Federated Learning
==================

Enable multiple parties to benefit from a machine learning model trained on their data
without handing over their data.

Federated Learning with Multiparty Computation (MPC) using the SPDZ protocol.

Concepts
--------

**Worker (aka Mine).** The worker is a server where private data is stored. The worker
downloads models and trains them using federated learning and differential privacy.

**Federated Learning.** Instead of bringing data all to one place for training, federated
learning is done by pushing the model to the data. This allows a data owner to maintain
the only copy of their information.

**Differential Privacy.** Differential Privacy is a set of techniques for preventing a
model from accidentally memorizing secrets present in a training dataset during the
learning process.

**Multi-party Computation (MPC).** When a model has multiple owners, multi-party computation
allows for individuals to share control of a model without seeing its contents such that
no sole owner can use or train it.

**Homomorphic Encryption.** When a model has a single owner, homomorphic encryption allows
an owner to encrypt their model so that untrusted 3rd parties can train or use the model
without being able to steal it.

**Transfer Learning.** Transfer learning is a machine learning method where a model developed
for a task (on one given set of data) is reused as the starting point for a model trained on
a second task (using another set of data).

**GAN Cryptography.** Uses generative adversarial neural networks (GANs) to protect
communications between two parties. Simpler alternative to homomorphic encryption.


Tangential Technologies of Interest
-----------------------------------

**IPFS.** IPFS (aka a global peer-to-peer merkle directed acyclic graph(1) file system) is a
distributed system for storing and accessing files, websites, applications, and data. Resiliency,
Decentralisation, Performance (like a CDN), Persistent Links.

(1) A `Merkle DAG <https://github.com/ipfs/specs/tree/master/merkledag>`_ is a data structure
similar to a Merkle Tree (as used to validate blockchain transactions) but not so strict. It
does not need to be balanced and its non-leaf nodes are allowed to contain data.

IPFS Notes:

* Same file contents have the same hash even if name is different.
* IPFS hashes always start with 'Qm'.
* IPFS supports source filtering and blacklisting; Swarm does not due to philosophical differences
  about potential censorship.
* Blockchain is not designed for storing a large amount of data or files.

Security:

Secure files can be encrypted with the public key of the intended recipient so that only they can
decrypt it when they retrieve it using IPFS. (The public key of the recipient is available via IPFS.)
`GPG <https://www.gnupg.org/>`_ can be used for the asymmetric encryption.

**Blockchain.** Blockchain can store a link to the encrypted data on IPFS. Blockchain is used to
validate integrity and completeness of the training data. (Could also be used for payments - setup a
market for data to create value for producers and therefore drive adoption rather than solely
relying on the dominant player in the marketplace, e.g. retailer.)


Companies & Organisations in the space
--------------------------------------

1. `OpenMined <https://www.openmined.org/>`_
2. `Effect.ai <https://effect.ai>`_ - Largely labelling, payment in crypto currency
3. `SingularityNET <https://singularitynet.io/>`_ - ditto


Homomorphic Encryption
----------------------

Libraries:

* `Microsoft SEAL <https://github.com/Microsoft/SEAL>`_ (C++, .NET)
  * `Microsoft/SEAL-Demo <https://github.com/Microsoft/SEAL-Demo>`_. Demos, Examples, Tutorials for using
    Microsoft SEAL library.
* `Lab41/PySEAL <https://github.com/Lab41/PySEAL>`_. This code wraps the Microsoft SEAL build in a docker
  container and provides a Python API to the encryption library.
  * `geworfener/using_pyseal <https://github.com/geworfener/using_pyseal>`_
* `HELib <https://github.com/homenc/HElib>`_. (C++) At its present state, this library is fairly low-level,
  and is best thought of as "assembly language for HE". That is, it provides low-level routines (set, add,
  multiply, shift, etc.), with as much access to optimizations as we can give. (Developed at IBM.)
* `vernamlab/cuFHE <https://github.com/vernamlab/cuFHE>`_. Low-level. Currently implemented gates are And,
  Or, Nand, Nor, Xor, Xnor, Not, Copy.
* `nucypher/nufhe <https://github.com/nucypher/nufhe>`_ (Python). This library implements the fully
  homomorphic encryption algorithm from TFHE using CUDA and OpenCL.
* `TFHE <https://tfhe.github.io/tfhe/>`_ (C++). The library supports the homomorphic evaluation of the
  10 binary gates (And, Or, Xor, Nand, Nor, etc…), as well as the negation and the Mux gate.
* `ibarrond/Pyfhel <https://github.com/ibarrond/Pyfhel>`_ (Python). Perform encrypted computations such as
  sum, mult, scalar product or matrix multiplication in Python, with NumPy compatibility. Uses SEAL, HElib,
  and PALISADE as backends. Implemented using Cython.
* `actuallyachraf/gomorph <https://github.com/actuallyachraf/gomorph>`_ (Go). Homomorphic Encryption in Golang.
  Implementation of `Paillier Cryptosystem <https://www.wikiwand.com/en/Paillier_cryptosystem>`_.
* `hardbyte/Paillier.jl <https://github.com/hardbyte/Paillier.jl>`_ (Julia). A Julia implementation of
  the Paillier partially homomorphic encryption system


GAN Cryptography
----------------

Implementations:

* `Numerai <https://numer.ai/>`_. Hedge fund using decentralised quant services.



References:

1. `Learn to securely share files on the blockchain with IPFS <https://medium.com/@mycoralhealth/learn-to-securely-share-files-on-the-blockchain-with-ipfs-219ee47df54c>`_
