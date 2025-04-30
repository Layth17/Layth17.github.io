---
layout: post
title: Introduction to Alibaba Cloud (ALIYUN)
description: starting point ...
categories:
- Tech
- Cloud
tags:
- cloud
- aliyun
date: 2025-04-28 19:58 -0500
---

> Reference: https://www.alibabacloud.com/help/en/ros/developer-reference/list-of-resource-types-by-service?spm=a2c63.p38356.help-menu-28850.d_5_1_1.38c72064WsyHRT


## Function Compute (FC)

```yaml
ROSTemplateFormatVersion: '2015-09-01'

Parameters:
  ServiceName:
    Type: String
    Description: Service name.

Resources:
  Functions:
    Type: DATASOURCE::FC::Functions
    Properties:
      ServiceName:
        Ref: ServiceName
```

## Resource Orchestration Service (ROS)

```yaml
ROSTemplateFormatVersion: '2015-09-01'
Description: Creates a machine user with basic developer permissions and access keys for terminal use

Resources:

  RamUser:
    Type: ALIYUN::RAM::User
    Properties:
      UserName: dev-machine-user
      PolicyAttachments:
        System: # 'System' basically says this role is managed for ya
          - AliyunDevsFullAccess

  RamUserAccessKey:
    Type: ALIYUN::RAM::AccessKey
    Properties:
      UserName: !GetAtt RamUser.UserName

  DevRole:
    Type: ALIYUN::RAM::Role
    Description: Dev role
    Properties:
      RoleName: dev-role
      MaxSessionDuration: 3600
      PolicyAttachments:
        System:
          - AliyunDevsFullAccess
      AssumeRolePolicyDocument:
        Version: '1'
        Statement:
          - Action: sts:AssumeRole
            Effect: Allow
            Principal:
              Service:
                - 'actiontrail.aliyuncs.com'
              ROS:
                - !Sub "acs:ram::${ALIYUN::AccountId}:root"

Outputs:
  UserId:
    Description: The RAM user ID created
    Value: !GetAtt RamUser.UserId
```