## MySQL 5.6 with Percona Xtrabackup
## taken from https://severalnines.com/database-blog/mysql-docker-building-container-image
 
## Pull the mysql:5.6 image
FROM mysql:5.6
 
## The maintainer name and email
MAINTAINER NieuweNils <email@notontheinternet.lol>
 
## List all packages that we want to install
ENV PACKAGE percona-xtrabackup-23
 
# Install requirement (wget)
RUN apt-get update && apt-get install -y wget
 
# Install Percona apt repository and Percona Xtrabackup
RUN wget https://repo.percona.com/apt/percona-release_0.1-3.jessie_all.deb && \
    dpkg -i percona-release_0.1-3.jessie_all.deb && \
    apt-get update && \
    apt-get install -y $PACKAGE
 
# Create the backup destination
RUN mkdir -p /backup/xtrabackups
 
# Allow mountable backup path
VOLUME ["/backup/xtrabackup"]
