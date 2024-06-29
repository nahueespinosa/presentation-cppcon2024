set -e

docker build -t cppcon2024-examples .
docker run -it --rm --cap-add=SYS_PTRACE --security-opt seccomp=unconfined cppcon2024-examples $@
