Vagrant Images
==============

This directory contains a few vagrant images which
can be used to build and develop on the latest Blaze
and its related dependencies. To get started, you'll
first want to install
[Vagrant](http://www.vagrantup.com/downloads.html)
and [VirtualBox](https://www.virtualbox.org/wiki/Downloads).

These images work on Linux, Mac OS X, and Windows, and
are an easy way to get a consistent, isolated environment
for Blaze development and experimentation.

Starting a Vagrant Image
------------------------

To provision and start one of these images, use
the following commands:

```
$ cd saucy64-py33
$ vagrant up
```

To connect with ssh, use:

```
$ vagrant ssh
```

On Windows,
[PuTTY](http://www.chiark.greenend.org.uk/~sgtatham/putty/download.html)
is a good way to ssh into these images. If you run
`vagrant ssh-config`, you will see output similar to
the following:

```
Host default
  HostName 127.0.0.1
  User vagrant
  Port 2222
  UserKnownHostsFile /dev/null
  StrictHostKeyChecking no
  PasswordAuthentication no
  IdentityFile C:/Users/[name]/.vagrant.d/insecure_private_key
  IdentitiesOnly yes
  LogLevel FATAL
```

You can use the PuTTYgen tool to convert the IdentityFile
private key from ssh format to the PuTTY .ppk format.
Then create a PuTTY configuration using the settings
and converted .ppk file.

Updating VirtualBox Guest Additions
-----------------------------------

You may find it useful to install the vagrant-vbguest
plugin, which automatically keeps the VirtualBox guest
additions up to date. When these are out of date,
features such as access to the `/vagrant` folder
may not work reliably. The command is:

```
$ vagrant plugin install vagrant-vbguest
```