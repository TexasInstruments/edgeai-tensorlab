DOCKER_CONFIG=$HOME/.docker
mkdir -p $DOCKER_CONFIG/cli-plugins

curl -SL https://github.com/docker/compose/releases/download/v2.12.2/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose

# Apply executable permissions to the binary:
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose

# Check version
docker compose version

