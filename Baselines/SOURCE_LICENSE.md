# Source and license

The files under `basicts/` are a source snapshot copied from the local
`DropoutTS` repository at commit `64a096e`. They retain the Apache-2.0 license
from `DropoutTS/LICENSE`; keep that license with any redistribution of the
snapshot. AdaWD-specific code lives outside this snapshot.

The WPMixer adapter is based on the MIT-licensed official repository
`Secure-and-Intelligent-Systems-Lab/WPMixer` at commit
`74104c9dddd54d279eb8323f48934b4fd75fcae7`. Its architecture was adapted to
the BasicTS configuration and tensor interfaces used here.

The TimeFilter and MultiPatchFormer official repositories did not contain a
software license at the reviewed commits. Their source is not redistributed
here. The corresponding packages are independent implementations of the
architectures described in the papers and expose the same BasicTS interfaces:

- `TROUBADOUR000/TimeFilter` at `dffde87e4fff0fdeeebbacde03dc1e432e15b3a1`;
- `bioinfoUQAM/MultiPatchFormer` at `965e6bd60822d509183253ef9c51fc3f9efe23f3`.
