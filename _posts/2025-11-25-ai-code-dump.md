---
layout: post
title: AI Code Dump
description: AI Code Dump -- Tread Carefully
categories:
- Tech
tags:
- Tech
date: 2025-11-25 15:58 -0500
---

- Fetch messages and their ID from AWS SQS (DLQ)

```py
import boto3
import json
from datetime import datetime, timedelta
import time

# Configuration
DLQ_URL = 'https://sqs.us-east-1.amazonaws.com/123456789/your-dlq-name'  # UPDATE THIS
HOURS_THRESHOLD = 24  # Look back this many hours
MAX_MESSAGES_TO_SCAN = 10000  # Safety limit
AWS_REGION = 'us-east-1'  # UPDATE THIS

def scan_dlq_for_recent_messages():
    """
    Scans a DLQ and collects MessageIds for messages sent within the threshold.
    """
    print('=== Starting DLQ Message ID Collection ===')
    print(f'DLQ URL: {DLQ_URL}')
    print(f'Time threshold: Last {HOURS_THRESHOLD} hours')
    print(f'Max messages to scan: {MAX_MESSAGES_TO_SCAN}')
    print(f'Region: {AWS_REGION}')
    
    # Initialize SQS client
    sqs = boto3.client('sqs', region_name=AWS_REGION)
    
    # Calculate cutoff time
    cutoff_time = datetime.now() - timedelta(hours=HOURS_THRESHOLD)
    cutoff_timestamp_ms = int(cutoff_time.timestamp() * 1000)
    print(f'Cutoff timestamp: {cutoff_time.isoformat()} ({cutoff_timestamp_ms}ms)')
    print('---\n')
    
    recent_message_ids = []
    recent_message_details = []
    total_messages_scanned = 0
    recent_messages_count = 0
    old_messages_count = 0
    
    try:
        has_more_messages = True
        empty_receive_count = 0
        max_empty_receives = 3
        
        while has_more_messages and total_messages_scanned < MAX_MESSAGES_TO_SCAN:
            # Receive messages from DLQ
            response = sqs.receive_message(
                QueueUrl=DLQ_URL,
                MaxNumberOfMessages=10,
                AttributeNames=['SentTimestamp', 'ApproximateReceiveCount', 'ApproximateFirstReceiveTimestamp'],
                MessageAttributeNames=['All'],
                VisibilityTimeout=30,  # Make invisible for 30 seconds while processing
                WaitTimeSeconds=2
            )
            
            messages = response.get('Messages', [])
            
            if not messages:
                empty_receive_count += 1
                print(f'No messages received (attempt {empty_receive_count}/{max_empty_receives})')
                
                if empty_receive_count >= max_empty_receives:
                    has_more_messages = False
                    print('Stopping: reached maximum empty receive attempts\n')
                continue
            
            empty_receive_count = 0
            print(f'Batch received: {len(messages)} messages')
            
            for message in messages:
                total_messages_scanned += 1
                
                # Extract message details
                sent_timestamp = int(message['Attributes']['SentTimestamp'])
                sent_date = datetime.fromtimestamp(sent_timestamp / 1000)
                message_id = message['MessageId']
                receive_count = message['Attributes']['ApproximateReceiveCount']
                
                # Calculate age
                age_hours = (datetime.now() - sent_date).total_seconds() / 3600
                
                # Check if message is recent
                is_recent = sent_timestamp >= cutoff_timestamp_ms
                
                if is_recent:
                    recent_messages_count += 1
                    recent_message_ids.append(message_id)
                    
                    message_detail = {
                        'messageId': message_id,
                        'sentTimestamp': sent_timestamp,
                        'sentDate': sent_date.isoformat(),
                        'ageInHours': round(age_hours, 2),
                        'receiveCount': receive_count,
                        'bodyPreview': message['Body'][:150] + ('...' if len(message['Body']) > 150 else ''),
                        'messageAttributes': message.get('MessageAttributes', {})
                    }
                    
                    recent_message_details.append(message_detail)
                    
                    print(json.dumps({
                        'type': 'RECENT_MESSAGE',
                        **message_detail
                    }, indent=2, default=str))
                    
                else:
                    old_messages_count += 1
                    
                    print(json.dumps({
                        'type': 'OLD_MESSAGE',
                        'messageId': message_id,
                        'sentDate': sent_date.isoformat(),
                        'ageInHours': round(age_hours, 2),
                        'receiveCount': receive_count
                    }, indent=2))
                
                # Return message to queue (make it visible again)
                sqs.change_message_visibility(
                    QueueUrl=DLQ_URL,
                    ReceiptHandle=message['ReceiptHandle'],
                    VisibilityTimeout=0
                )
            
            # Progress update
            if total_messages_scanned % 50 == 0:
                print(f'\nProgress: Scanned {total_messages_scanned} messages...\n')
            
            # Small delay to avoid throttling
            time.sleep(0.1)
        
        # Print summary
        print('\n---')
        print('=== SCAN COMPLETE ===')
        print(f'Total messages scanned: {total_messages_scanned}')
        print(f'Recent messages (within {HOURS_THRESHOLD}h): {recent_messages_count}')
        print(f'Old messages (older than {HOURS_THRESHOLD}h): {old_messages_count}')
        print('---\n')
        
        # Print MessageIds list
        print('=== RECENT MESSAGE IDS ===')
        print(json.dumps(recent_message_ids, indent=2))
        print('---\n')
        
        # Print detailed information
        print('=== RECENT MESSAGE DETAILS ===')
        print(json.dumps(recent_message_details, indent=2, default=str))
        print('---\n')
        
        # Save to files
        output_data = {
            'summary': {
                'totalScanned': total_messages_scanned,
                'recentCount': recent_messages_count,
                'oldCount': old_messages_count,
                'hoursThreshold': HOURS_THRESHOLD,
                'cutoffDate': cutoff_time.isoformat(),
                'scanDate': datetime.now().isoformat()
            },
            'recentMessageIds': recent_message_ids,
            'recentMessageDetails': recent_message_details
        }
        
        # Save MessageIds to a simple text file
        with open('recent_message_ids.txt', 'w') as f:
            for msg_id in recent_message_ids:
                f.write(f'{msg_id}\n')
        print('✓ Saved MessageIds to: recent_message_ids.txt')
        
        # Save full details to JSON
        with open('recent_messages_full.json', 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print('✓ Saved full details to: recent_messages_full.json')
        
        return output_data
        
    except Exception as error:
        print('\n=== ERROR ===')
        print(f'Error processing DLQ messages: {error}')
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    try:
        result = scan_dlq_for_recent_messages()
        print('\n=== SUCCESS ===')
        print(f"Found {result['summary']['recentCount']} recent messages")
        print('Check the output files for the complete list of MessageIds')
    except KeyboardInterrupt:
        print('\n\nScan interrupted by user')
    except Exception as e:
        print(f'\n\nFailed to complete scan: {e}')
```