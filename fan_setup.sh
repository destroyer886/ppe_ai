cat << 'EOF' > /usr/local/bin/jetson_init.sh
#!/bin/bash
# Set Fan to Max
if [ -e /sys/devices/pwm-fan/target_pwm ]; then
    echo 255 > /sys/devices/pwm-fan/target_pwm
fi

# Set Static IP (Replace eth0 with your interface name)
# nmcli con mod "Wired connection 1" ipv4.addresses 192.168.1.100/24 ipv4.method manual
# nmcli con up "Wired connection 1"
EOF


