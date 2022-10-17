#!/bin/bash

WIFI_IP=$(ip addr show | grep wlan0 | grep inet | awk '{print $2}')
DEFAULT_ROUTE=$(ip route show | grep default | awk '{print $3}')
NAME_SERVERS=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}' | tr '\n' ' ')


sudo apt-get update
sudo apt-get -y upgrade
sudo sed -i 's/rootwait/cgroup_memory=1 cgroup_enable=memory rootwait/' /boot/cmdline.txt
ssh-keygen -N "" -f /home/pi/.ssh/id_rsa
sudo apt-get -y install vim screen

if ( ! grep -q "interface wlan0" /etc/dhcpcd.conf ); then

sudo cat << EOF >> /etc/dhcpcd.conf
interface wlan0
static ip_address=${WIFI_IP}
static routers=${DEFAULT_ROUTE}
static domain_name_servers=${NAME_SERVERS}

interface eth0
static ip_address=192.168.99.1
EOF

fi

cat << 'EOF' > /tmp/startup
WIFI_IP=$(ip addr show | grep wlan0 | grep inet | awk '{print $2}' | awk -F '/' '{print $1}')
iptables -t nat -A POSTROUTING -s 192.168.99.0/24 -j SNAT --to ${WIFI_IP}
exit 0
EOF

if ( ! grep -q WIFI_IP /etc/rc.local ); then
sudo sed -i '/^exit 0/ {
    r /tmp/startup
    d
}' /etc/rc.local
fi

curl -sfL https://get.k3s.io | sh -