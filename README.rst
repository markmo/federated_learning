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

Security:

Secure files can be encrypted with the public key of the intended recipient so that only they can
decrypt it when they retrieve it using IPFS. (The public key of the recipient is available via IPFS.)
`GPG <https://www.gnupg.org/>`_ can be used for the asymmetric encryption.

**Blockchain.** Blockchain can store a link to the encrypted data on IPFS. Blockchain is used to
validate integrity and completeness of the training data. (Could also be used for payments - setup a
market for data to create value for producers, and therefore drive adoption rather than solely
relying on marketplace dominant players, e.g. retailer.)


Companies & Organisations in the space
--------------------------------------

1. `OpenMined <https://www.openmined.org/>`_
2. `Effect.ai <https://effect.ai>`_ - Largely labelling, payment in crypto currency
3. `SingularityNET <https://singularitynet.io/>`_ - ditto


References:

1. `Learn to securely share files on the blockchain with IPFS <https://medium.com/@mycoralhealth/learn-to-securely-share-files-on-the-blockchain-with-ipfs-219ee47df54c>`_
