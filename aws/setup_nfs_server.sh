#!/usr/bin/env bash

if [ $# -eq 0 ]; then
        echo "Usage: $0 <new_user> [new_user_uid]"
        exit 1
fi

sudo apt install nfs-kernel-server -y

sudo mkdir /srv/nfs
sudo chmod 777 /srv/nfs
sudo chown nobody:nogroup /srv/nfs
mkdir /srv/nfs/$USER

# update /etc/exports
echo "/srv/nfs           *(rw,async,no_root_squash,no_subtree_check)" | sudo tee -a /etc/exports

# start the nfs server
sudo systemctl restart rpcbind.service
sudo systemctl restart nfs-server.service

# for all clients, run this
sudo mkdir /nfs
sudo chmod 777 /nfs
sudo chown nobody:nogroup /nfs

ln -sf /nfs/$USER $HOME/nfs

sudo mount -t nfs4 -o tcp,timeo=1,retrans=3 a8:/srv/nfs /nfs -v