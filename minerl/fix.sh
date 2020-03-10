#!/bin/bash
sudo apt-get purge openjdk-8*


sudo apt-get install openjdk-8-jre-headless=8u162-b12-1
sudo apt-get install openjdk-8-jdk-headless=8u162-b12-1
sudo apt-get install openjdk-8-jre=8u162-b12-1
sudo apt-get install openjdk-8-jdk=8u162-b12-1

sudo update-java-alternatives -s java-1.8.0-openjdk-amd64
