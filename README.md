Dataset Download
LLaVA-665K dataset is available on Download Link.

Cambrian-7M dataset is available on Download Link.

Then follow the original repo to download the image data.

You can split the data into random chunks for parallel gradient computation using slurm scripts. For efficient processing, request as many CPUs as possible (e.g., 96 CPUs), as the splitting operation is CPU-intensive and can be parallelized. For example to split the 7M Cambrian dataset into 3000 chunks with 96 CPUs takes about 10-15 minutes.
