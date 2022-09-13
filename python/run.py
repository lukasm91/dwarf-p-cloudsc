#!/usr/bin/env python3
import io
import math
import h5py
import argparse
import numpy as np
import types


class FortranWriter:
    def __init__(self, path):
        self.file = open(path, "wb")

    def write(self, arr):
        self.file.write(np.array([0], dtype="<i4").tobytes())
        self.file.write(np.array(list(reversed(arr.shape)), dtype="<i4").tobytes())
        self.file.write(np.array([0], dtype="<i4").tobytes())
        self.file.write(arr.tobytes())
        self.file.write(np.array([0], dtype="<i4").tobytes())


def load_params(inp, keys=None, prefix=None):
    ret = dict()
    if prefix is not None:
        assert keys is None
        for k in inp.keys():
            if k.lower().startswith(prefix.lower()):
                ret[k[len(prefix) :].lower()] = inp[k][0]
    else:
        assert keys is not None
        assert prefix is None
        for k in keys:
            ret[k.lower()] = inp[k.upper()][0]
    return types.SimpleNamespace(**ret)


def block_field(arr, ngptot, nproma, dtype):
    nblocks = (ngptot + nproma - 1) // nproma

    arr = np.moveaxis(arr, source=-1, destination=0)
    assert arr.shape[0] == 100, "Unexpected shape"
    nreps = (nblocks * nproma + arr.shape[0] - 1) // arr.shape[0]
    arr = np.tile(arr, (nreps, *np.ones(arr.ndim - 1, dtype=int)))
    arr = arr[: nblocks * nproma, ...]
    arr[ngptot:, ...] = 0
    arr = arr.reshape((nblocks, nproma, *arr.shape[1:]))
    arr = np.moveaxis(arr, source=1, destination=-1).astype(dtype)

    return arr.copy()


def load_tendencies(inp, name, ngptot, nproma, dtype):
    t = block_field(inp[f"{name}_T"][()], nproma=nproma, ngptot=ngptot, dtype=dtype)
    a = block_field(inp[f"{name}_Q"][()], nproma=nproma, ngptot=ngptot, dtype=dtype)
    q = block_field(inp[f"{name}_A"][()], nproma=nproma, ngptot=ngptot, dtype=dtype)
    cld = block_field(inp[f"{name}_CLD"][()], nproma=nproma, ngptot=ngptot, dtype=dtype)

    t = np.expand_dims(t, 1)
    a = np.expand_dims(a, 1)
    q = np.expand_dims(q, 1)

    ret = np.concatenate([t, a, q, cld], axis=1)
    return ret


def read_inputs(ngptot, nproma, dtype):
    fields = {}
    outputs = {}
    scalars = {}
    constants = {}

    with h5py.File("/work/run/input.h5", "r") as f:
        constants["yomcst"] = load_params(
            f, keys=["rg", "rd", "rcpd", "retv", "rlvtt", "rlstt", "rlmlt", "rtt", "rv"]
        )
        constants["yoethf"] = load_params(
            f,
            keys=[
                "r2es",
                "r3les",
                "r3ies",
                "r4les",
                "r4ies",
                "r5les",
                "r5ies",
                "r5alvcp",
                "r5alscp",
                "ralvdcp",
                "ralsdcp",
                "ralfdcp",
                "rtwat",
                "rtice",
                "rticecu",
                "rtwat_rtice_r",
                "rtwat_rticecu_r",
                "rkoop1",
                "rkoop2",
            ],
        )
        constants["yrecldp"] = load_params(f, prefix="yrecldp_")
        constants["yrephli"] = load_params(f, prefix="yrephli_")
        constants = types.SimpleNamespace(**constants)

        klev = f["KLEV"][0]

        fields["plcrit_aer"] = block_field(
            f["PLCRIT_AER"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["picrit_aer"] = block_field(
            f["PICRIT_AER"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["pre_ice"] = block_field(
            f["PRE_ICE"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["pccn"] = block_field(
            f["PCCN"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["pnice"] = block_field(
            f["PNICE"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["pt"] = block_field(
            f["PT"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["pq"] = block_field(
            f["PQ"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        #  fields["pvfa"] = block_field(f["PVFA"][()], nproma=nproma, ngptot=ngptot)
        fields["pvfl"] = block_field(
            f["PVFL"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["pvfi"] = block_field(
            f["PVFI"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        #  fields["pdyna"] = block_field(f["PDYNA"][()], nproma=nproma, ngptot=ngptot)
        #  fields["pdynl"] = block_field(f["PDYNL"][()], nproma=nproma, ngptot=ngptot)
        #  fields["pdyni"] = block_field(f["PDYNI"][()], nproma=nproma, ngptot=ngptot)
        fields["phrsw"] = block_field(
            f["PHRSW"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["phrlw"] = block_field(
            f["PHRLW"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["pvervel"] = block_field(
            f["PVERVEL"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["pap"] = block_field(
            f["PAP"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["paph"] = block_field(
            f["PAPH"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["plsm"] = block_field(
            f["PLSM"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["ldcum"] = block_field(
            f["LDCUM"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["ktype"] = block_field(
            f["KTYPE"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["plu"] = block_field(
            f["PLU"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["plude"] = block_field(
            f["PLUDE"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["psnde"] = block_field(
            f["PSNDE"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["pmfu"] = block_field(
            f["PMFU"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["pmfd"] = block_field(
            f["PMFD"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["pa"] = block_field(
            f["PA"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["pclv"] = block_field(
            f["PCLV"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["psupsat"] = block_field(
            f["PSUPSAT"][()], nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        #  ! Note: The 0-sized array (KFLDX=0) seems to create problems when filled with
        #  ! data from the C-backend, causing memory corruption if enabled.
        #  fields["pextra"] = block_field(f["PEXTRA"][()], nproma=nproma, ngptot=ngptot)

        fields["tendency_tmp"] = load_tendencies(
            f, "TENDENCY_TMP", nproma=nproma, ngptot=ngptot, dtype=dtype
        )
        fields["tendency_tmp_t"] = fields["tendency_tmp"][:, 0, :, :]
        fields["tendency_tmp_q"] = fields["tendency_tmp"][:, 1, :, :]
        fields["tendency_tmp_a"] = fields["tendency_tmp"][:, 2, :, :]
        fields["tendency_tmp_cld"] = fields["tendency_tmp"][:, 3:, :, :]
        fields["tendency_loc"] = np.zeros_like(fields["tendency_tmp"])
        fields["tendency_loc_t"] = fields["tendency_loc"][:, 0, :, :]
        fields["tendency_loc_q"] = fields["tendency_loc"][:, 1, :, :]
        fields["tendency_loc_a"] = fields["tendency_loc"][:, 2, :, :]
        fields["tendency_loc_cld"] = fields["tendency_loc"][:, 3:, :, :]
        fields = types.SimpleNamespace(**fields)

        scalars["ptsphy"] = f["PTSPHY"][0]
        scalars["ldslphy"] = f["LDSLPHY"][0]
        #  scalars["ldmaincall"] = f["LDMAINCALL"][0]
        scalars = types.SimpleNamespace(**scalars)

    outputs["prainfrac_toprfz"] = block_field(
        np.zeros((100,)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs["pcovptot"] = block_field(
        np.zeros((klev, 100)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs["pfsqlf"] = block_field(
        np.zeros((klev + 1, 100)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs["pfsqif"] = block_field(
        np.zeros((klev + 1, 100)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs["pfcqlng"] = block_field(
        np.zeros((klev + 1, 100)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs["pfcqnng"] = block_field(
        np.zeros((klev + 1, 100)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs["pfsqrf"] = block_field(
        np.zeros((klev + 1, 100)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs["pfsqsf"] = block_field(
        np.zeros((klev + 1, 100)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs["pfcqrng"] = block_field(
        np.zeros((klev + 1, 100)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs["pfcqsng"] = block_field(
        np.zeros((klev + 1, 100)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs["pfsqltur"] = block_field(
        np.zeros((klev + 1, 100)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs["pfsqitur"] = block_field(
        np.zeros((klev + 1, 100)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs["pfplsl"] = block_field(
        np.zeros((klev + 1, 100)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs["pfplsn"] = block_field(
        np.zeros((klev + 1, 100)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs["pfhpsl"] = block_field(
        np.zeros((klev + 1, 100)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs["pfhpsn"] = block_field(
        np.zeros((klev + 1, 100)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs["tmp1"] = block_field(
        np.zeros((1, klev + 1, 100)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs["tmp2"] = block_field(
        np.zeros((1, klev + 1, 100)), nproma=nproma, ngptot=ngptot, dtype=dtype
    )
    outputs = types.SimpleNamespace(**outputs)

    return klev, constants, fields, scalars, outputs


parser = argparse.ArgumentParser(description="Run CLOUDSC dwarf")
parser.add_argument("--ngptot", type=int, help="number of columns", default=163840)
parser.add_argument("--nproma", type=int, help="blocking parameter", default=128)
parser.add_argument("--dtype", type=str, help="data type", default="double")
parser.add_argument("--config", type=int, help="configuration", default=1)
parser.add_argument("--verify", action="store_true")
run_args = parser.parse_args()
ngptot = run_args.ngptot
nproma = run_args.nproma
nblocks = (ngptot + nproma - 1) // nproma

if run_args.dtype == "single":
    dtype = np.float32
elif run_args.dtype == "double":
    dtype = np.float64
else:
    assert False, "Unknown data type"

klev, c, f, s, o = read_inputs(nproma=nproma, ngptot=ngptot, dtype=dtype)

p = ""
for g, vs in c.__dict__.items():
    p += "namespace {"
    for v, vv in vs.__dict__.items():
        if type(vv) is np.bool_:
            p += f"constexpr bool {v} = {'true' if vv else 'false'};"
        elif type(vv) is np.float64:
            p += f"constexpr real_t {v} = {vv};"
        elif type(vv) is np.int64:
            p += f"constexpr int {v} = {vv};"
        else:
            print(type(vv))
            assert False
    p += "}\n"


import cloudsc_cuda

cloudsc_cuda.run(
    klev,
    ngptot,
    nproma,
    c.__dict__,
    f.__dict__,
    s.__dict__,
    o.__dict__,
    run_args.config,
    not run_args.verify,
)

#  import python_impl
#  python_impl.run_cloud_scheme(klev, ngptot, nproma, c, f, s, o, dtype)

if run_args.verify:
    out = FortranWriter("serialized_gpu_cuda.dat")

    out.write(f.plude)
    out.write(o.pcovptot)
    out.write(o.prainfrac_toprfz)
    out.write(o.pfsqlf)
    out.write(o.pfsqif)
    out.write(o.pfcqlng)
    out.write(o.pfcqnng)
    out.write(o.pfsqrf)
    out.write(o.pfsqsf)
    out.write(o.pfcqrng)
    out.write(o.pfcqsng)
    out.write(o.pfsqltur)
    out.write(o.pfsqitur)
    out.write(o.pfplsl)
    out.write(o.pfplsn)
    out.write(o.pfhpsl)
    out.write(o.pfhpsn)
    f.tendency_tmp[:, [1, 2], :, :] = f.tendency_tmp[:, [2, 1], :, :]
    f.tendency_loc[:, [1, 2], :, :] = f.tendency_loc[:, [2, 1], :, :]
    out.write(f.tendency_tmp)
    out.write(f.tendency_loc)
