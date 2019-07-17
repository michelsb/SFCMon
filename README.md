# SFCMon

## Introduction

Repository SFCMon is an efficient and scalable monitoring solution to keep track network  ows in SFC-enabled domains.

## Software Requirements

The virtualization software is portable and should run on a few OSes:

  * Linux
  * Windows PowerShell 3 or later
  * FreeBSD
  * Mac OS X
  * Solaris

## Obtaining required software

The following applications are required:

  * Vagrant: https://www.vagrantup.com/downloads.html
  * VirtualBox: https://www.virtualbox.org/wiki/Downloads

You will need to build a virtual machine. For this, follow the steps below:

 1. Install VirtualBox;
 2. Install Vagrant (use the site installer even on Linux);
 3. Install Vagrant plugins:
    `vagrant plugin install vagrant-disksize`
 4. Download or clone the SFCMon repository: 
    `git clone -b develop https://github.com/michelsb/SFCMon.git`
 5. Access the folder and create the VM:
    `cd SFCMon/create-dev-env`
    `vagrant up`
 
Other auxiliary commands:

- Accessing the VM: `vagrant ssh`
- Halt the VM: `vagrant halt` (outside VM) or `sudo shutdown -h now` (inside VM)
- Destroy the VM: `vagrant destroy

## Usage

