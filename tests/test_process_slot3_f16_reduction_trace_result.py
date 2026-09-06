import hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
from scripts.instrument_slot3_f16_reduction_trace import REGISTERS
from scripts.process_slot3_f16_reduction_trace_result import TRACE_BYTES,UNINSTRUMENTED_OUTPUT_SHA256,process_archive
class TestReductionReceiver(unittest.TestCase):
    def make(self,root,traversal=False):
        t=bytearray(TRACE_BYTES);t[-1]=1;t=bytes(t);m={'schema':1,'experiment':'rtx5070_slot3_f16_reduction_register_trace','status':'PASS','payload_integrity':True,'probe_exit':0,'probe_pass':True,'device_name':'NVIDIA GeForce RTX 5070','cta':[2,0,0],'registers':REGISTERS,'kernel_launched':True,'checkpoint_bytes':TRACE_BYTES,'checkpoint_nonzero_bytes':1,'checkpoint_sha256':hashlib.sha256(t).hexdigest().upper(),'output_sha256':UNINSTRUMENTED_OUTPUT_SHA256};z=root/'x.zip'
        with zipfile.ZipFile(z,'w') as f:f.writestr('_r/manifest.json',json.dumps(m));f.writestr('_r/f16_reduction_trace.raw',t);f.writestr('../bad','x') if traversal else None
        return z,t
    def test_valid(self):
        with tempfile.TemporaryDirectory() as d:r=Path(d);z,t=self.make(r);a=r/'a';a.write_bytes(t);receipt=process_archive(z,a,r/'o');self.assertTrue(receipt['bitwise_equal']);self.assertTrue(receipt['admissible_as_uninstrumented_oracle'])
    def test_traversal(self):
        with tempfile.TemporaryDirectory() as d:r=Path(d);z,t=self.make(r,True);a=r/'a';a.write_bytes(t);self.assertRaisesRegex(ValueError,'unsafe ZIP member',process_archive,z,a,r/'o')
if __name__=='__main__':unittest.main()
