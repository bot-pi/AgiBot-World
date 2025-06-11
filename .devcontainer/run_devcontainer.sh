# run docker container inside a ubuntu server to support devcontainer
#!/bin/bash
docker run \
    -it \
    --rm \
    -v $(pwd):/workspace \
    agibotworld:test-server