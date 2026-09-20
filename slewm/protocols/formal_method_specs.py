"""Frozen method axes shared by formal training evaluation and comparison."""

FORMAL_METHOD_SIGNATURES = {
    "champion": {
        "idm_weight": 0.0,
        "idm_input_mode": "concat",
        "dcl_idm_weight": 0.05,
        "ibr_schedule_v1": False,
    },
    "ibr": {
        "idm_weight": 0.0,
        "idm_input_mode": "concat",
        "dcl_idm_weight": 0.05,
        "ibr_schedule_v1": True,
    },
    "concat_mse": {
        "idm_weight": 0.03,
        "idm_input_mode": "concat",
        "dcl_idm_weight": 0.0,
        "ibr_schedule_v1": False,
    },
    "delta_mse": {
        "idm_weight": 0.03,
        "idm_input_mode": "delta",
        "dcl_idm_weight": 0.0,
        "ibr_schedule_v1": False,
    },
}


# Exact training-only state carried by a transition-triangle checkpoint.  Keep
# this lightweight contract here so both formal evaluation and short engineering
# smokes reject missing, extra, wrong-shaped, or wrong-dtype auxiliary tensors.
TRAINING_AUXILIARY_PREFIXES = (
    "head_f.",
    "head_g.",
    "head_f_idm.",
    "head_g_idm.",
    "idm_mlp.",
    "sigreg_module.",
)

COMMON_AUXILIARY_SHAPES = {
    "head_f.net.0.weight": (384, 384),
    "head_f.net.0.bias": (384,),
    "head_f.net.1.weight": (384,),
    "head_f.net.1.bias": (384,),
    "head_f.net.3.weight": (64, 384),
    "head_f.net.3.bias": (64,),
    "head_g.net.0.weight": (256, 192),
    "head_g.net.0.bias": (256,),
    "head_g.net.1.weight": (256,),
    "head_g.net.1.bias": (256,),
    "head_g.net.3.weight": (64, 256),
    "head_g.net.3.bias": (64,),
    "sigreg_module.phi": (17,),
    "sigreg_module.t": (17,),
    "sigreg_module.weights": (17,),
}

DCL_IDM_AUXILIARY_SHAPES = {
    "head_f_idm.0.weight": (256, 384),
    "head_f_idm.0.bias": (256,),
    "head_f_idm.1.weight": (256,),
    "head_f_idm.1.bias": (256,),
    "head_f_idm.3.weight": (64, 256),
    "head_f_idm.3.bias": (64,),
    "head_g_idm.0.weight": (64, 10),
    "head_g_idm.0.bias": (64,),
    "head_g_idm.1.weight": (64,),
    "head_g_idm.1.bias": (64,),
    "head_g_idm.3.weight": (64, 64),
    "head_g_idm.3.bias": (64,),
}

MSE_IDM_AUXILIARY_SHAPES = {
    "concat": {
        "idm_mlp.0.weight": (256, 384),
        "idm_mlp.0.bias": (256,),
        "idm_mlp.2.weight": (10, 256),
        "idm_mlp.2.bias": (10,),
    },
    "delta": {
        "idm_mlp.0.weight": (256, 192),
        "idm_mlp.0.bias": (256,),
        "idm_mlp.2.weight": (10, 256),
        "idm_mlp.2.bias": (10,),
    },
}


def formal_auxiliary_shapes(method_name):
    """Return the exact float32 auxiliary state required by one formal arm."""
    if method_name not in FORMAL_METHOD_SIGNATURES:
        raise KeyError(f"unknown formal method: {method_name}")
    expected = dict(COMMON_AUXILIARY_SHAPES)
    if method_name in {"champion", "ibr"}:
        expected.update(DCL_IDM_AUXILIARY_SHAPES)
    else:
        input_mode = FORMAL_METHOD_SIGNATURES[method_name]["idm_input_mode"]
        expected.update(MSE_IDM_AUXILIARY_SHAPES[input_mode])
    return expected
