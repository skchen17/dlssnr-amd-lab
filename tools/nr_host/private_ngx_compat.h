// ===========================================================================
// private_ngx_compat.h — PRIVATE ABI COMPATIBILITY LAYER (dlssnr-amd-lab)
// ===========================================================================
// PRIVATE ABI — inferred. NOT part of NVIDIA's public NGX ABI.
//
// Everything in this header is *inferred* from the open-source DLSS5-Feeder
// project (MIT, jlrouzies-fr/DLSS5-Feeder @ 80abd23fa41a) and leaked-runtime
// observation. Nothing here is declared by the official NGX headers in
// third_party/nvidia-dlss/include/, where feature ids 14..18 are explicitly
// `NVSDK_NGX_Feature_Reserved*` placeholders.
//
// RULES:
//   1. The public (official) NGX ABI lives ONLY in the official headers;
//      nr_host.cpp includes them directly. Never re-declare official symbols
//      here.
//   2. Feature id 18 may appear ONLY in this file and at its single use site
//      in nr_host.cpp (Stage B). Every artifact produced through it must be
//      labeled "PRIVATE_ABI".
//   3. Do not use this layer in Stage A (vanilla DLSS/DLAA). Stage A uses
//      NVSDK_NGX_Feature_SuperSampling only.
// ===========================================================================

#ifndef DLSSNR_PRIVATE_NGX_COMPAT_H
#define DLSSNR_PRIVATE_NGX_COMPAT_H

#include "nvsdk_ngx.h"  // official public ABI (third_party/nvidia-dlss/include)

// PRIVATE ABI — inferred: DLSS5 runtime feature id for Neural Rendering.
// Source: DLSS5-Feeder host logs ("feature 18 created", "inline feature 18
// evaluation succeeded"). In the official enum this value is
// NVSDK_NGX_Feature_Reserved18. There is NO public declaration of feature 18.
#define PRIVATE_NGX_FEATURE_NEURAL_RENDERING ((NVSDK_NGX_Feature)18)

// PRIVATE ABI — inferred: evaluation of feature 18 is expected to happen
// *inline* inside the DLSS evaluate pass when the DLSS5 addon has armed its
// hook (see DLSS5-Feeder config: create_delay, warmup_rebuild). The host must
// therefore route feature 18 attempts through the same official
// CreateFeature/EvaluateFeature entry points — there are no separate private
// entry points known to this lab.
static inline const char* PrivateNgxFeatureName(NVSDK_NGX_Feature f) {
    if (f == PRIVATE_NGX_FEATURE_NEURAL_RENDERING) return "NeuralRendering(feature18, PRIVATE_ABI)";
    return nullptr;
}

#endif // DLSSNR_PRIVATE_NGX_COMPAT_H
