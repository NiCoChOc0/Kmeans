with import <nixpkgs> {};

(pkgs.python3.withPackages (ps: with ps; [
ipykernel jupyterlab
matplotlib numpy scikit-learn pandas scikit-image
networkx
])).env
