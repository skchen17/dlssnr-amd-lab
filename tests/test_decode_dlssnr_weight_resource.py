import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

from scripts.decode_dlssnr_weight_resource import ALIGNMENT, decode_resource, parse_resource


def make_record(name: str, payload: bytes) -> bytes:
    encoded = name.encode("ascii")
    if len(payload) % 2:
        raise ValueError("test FP16 payload must have an even byte count")
    packed_bytes = len(payload) + 40
    return b"".join(
        (
            struct.pack("<Q", len(encoded)),
            encoded,
            struct.pack("<QQQI", packed_bytes, packed_bytes, len(payload), 1),
            payload,
            struct.pack("<IIIII", 0, 0, 1, 0, len(payload) // 2),
        )
    )


def make_resource(records: list[tuple[str, bytes]]) -> bytes:
    body = b"".join(make_record(name, payload) for name, payload in records)
    return struct.pack("<Q", len(body) + 8) + body


class WeightResourceDecodeTests(unittest.TestCase):
    def test_decodes_named_fp16_records_to_aligned_arena(self):
        resource = make_resource([("block0.layer0.layer", b"\x01\x02\x03\x04"),
                                  ("block1.layer0.layer", b"\x05\x06")])
        records = parse_resource(resource)
        self.assertEqual(["block0.layer0.layer", "block1.layer0.layer"],
                         [record.name for record in records])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resource_path = root / "weights_resource.bin"
            resource_path.write_bytes(resource)
            expected = b"\x01\x02\x03\x04" + b"\0" * (ALIGNMENT - 4)
            expected += b"\x05\x06" + b"\0" * (ALIGNMENT - 2)
            reference_path = root / "reference.raw"
            reference_path.write_bytes(expected)
            manifest = decode_resource(resource_path, root / "decoded", reference_path)

            self.assertEqual("PASS", manifest["status"])
            self.assertTrue(manifest["reference"]["exact_match"])
            self.assertEqual(hashlib.sha256(expected).hexdigest().upper(),
                             manifest["model_arena"]["sha256"])
            self.assertEqual(expected, (root / "decoded" / "model_arena.raw").read_bytes())
            self.assertEqual([0, ALIGNMENT],
                             [tensor["arena_offset"] for tensor in manifest["tensors"]])

    def test_rejects_invalid_declared_size(self):
        resource = bytearray(make_resource([("block0.layer0.layer", b"\x00\x00")]))
        struct.pack_into("<Q", resource, 0, len(resource) + 1)
        with self.assertRaisesRegex(ValueError, "resource size header"):
            parse_resource(bytes(resource))


if __name__ == "__main__":
    unittest.main()
