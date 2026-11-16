# Third-Party Software And Data

The MIT license covers the original code in this repository, not third-party
datasets, model weights, or dependencies.

- RSPD uses [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) through
  [OpenCLIP](https://github.com/mlfoundations/open_clip). RemoteCLIP weights
  are downloaded from their original Hugging Face repository, not redistributed
  here. The pinned revision and checksum are in the bundled provenance JSON.
- Aligned Attention Score (AAS) is adapted from
  [DiffSim](https://github.com/showlab/DiffSim), by Yiren Song, Xiaokang Liu,
  and Mike Zheng Shou, "DiffSim: Taming Diffusion Models for Evaluating Visual
  Similarity," arXiv:2412.14580. The optional DiffSim baseline uses a separately
  obtained upstream checkout; no upstream DiffSim source is vendored here.
- Optional [LPIPS](https://github.com/richzhang/PerceptualSimilarity) and
  [DreamSim](https://github.com/ssundaram21/dreamsim) baselines are installed
  as dependencies and retain their respective licenses.
- Benchmark inputs come from RSICD/RSIC-H, VEDAI, EarthView/Sentinel-2, and
  SAR-to-optical source data. See `docs/DATA.md`. A source being publicly
  downloadable does not establish permission to redistribute derived images.

Please cite the corresponding works when using their methods or datasets.
