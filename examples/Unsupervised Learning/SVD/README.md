# Singular Value Decomposition (SVD) for Image Compression

In the attached notebook `svd_image_compression.ipynb`, we explore **Singular Value Decomposition (SVD)**, a fundamental matrix factorization technique in linear algebra with powerful applications in data science. At its core, SVD decomposes a matrix into singular vectors and singular values, which represent the directions and magnitude of variance in the data.

SVD is particularly useful for **dimensionality reduction** and **low-rank approximations**, where we can retain the most important features of data while discarding minor details or noise. This makes it ideal for tasks such as image compression, noise reduction, and latent feature extraction.

We apply SVD to an image of a squirrel (`squirrel_pic.jpeg`) to illustrate how the technique can reduce storage requirements while preserving visual information. In the notebook, we explore the math, the intuition, and the practical trade-offs involved in choosing the number of singular values to retain.

---

**In this project, we**:

* Load and preprocess an image, converting it to grayscale to simplify analysis.

* Apply SVD to decompose the image into **U**, **Σ**, and **V^T** matrices, highlighting how singular values capture the most important structures in the image.

* Reconstruct the image using varying numbers of singular values (**rank \$k\$**), demonstrating how low-rank approximations retain essential features while reducing storage.

* Compute and visualize **reconstruction error** using the Frobenius norm, showing how image quality improves as more singular values are included.

* Explore **energy-based rank selection**, analyzing cumulative variance to choose the number of singular values needed to retain a desired percentage of the image’s information.

* Discuss practical applications of SVD beyond image compression, including dimensionality reduction, noise filtering, latent semantic analysis, and more.

This notebook is intended as a hands-on introduction to SVD, providing both a conceptual understanding of matrix decomposition and a practical demonstration of its use in compressing and analyzing real-world data.


