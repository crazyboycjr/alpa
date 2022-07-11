#!/bin/bash

# mount shared directory
sudo mount -t nfs4 -o tcp,timeo=1,retrans=3 a8:/srv/nfs /nfs -v