# Grouping cache

`grouping_cache.pt` is 172,432,625 bytes, so it is stored as two parts below GitHub's 100 MB file limit. Reassemble it in this directory before reusing the cached grouping and saliency data:

```sh
cat grouping_cache.pt.part00 grouping_cache.pt.part01 > grouping_cache.pt
shasum -a 256 grouping_cache.pt
```

Expected SHA-256: `0ace5ae6b7ccfa2a87677d5bee38c21512c1a95116ec2321c9d7da7d7dfeb061`.
