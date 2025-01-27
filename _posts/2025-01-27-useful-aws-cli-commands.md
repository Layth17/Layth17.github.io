---
layout: post
title: Useful AWS CLI Commands
description: commands I have found to be useful
categories:
- Tech
- Cloud
tags:
- cloud
- aws
date: 2025-01-27 00:58 -0500
---

## SSM

- make sure you can `ping` host within the ec2 instance or use `telnet`

```bash
aws ssm start-session --target ec2-instance-id \
--document-name AWS-StartPortForwardingSessionToRemoteHost \
--parameters host="your-host",portNumber="host-port",localPortNumber="your-port"
```

- example for something like Aurora MySQL Cluster

```bash
aws ssm start-session --target i-######## \
--document-name AWS-StartPortForwardingSessionToRemoteHost \
--parameters host="name.cluster-randomchars.region.rds.amazonaws.com",portNumber="3306",localPortNumber="9000"
```