CREATE KEYSPACE IF NOT EXISTS jarvis_test
WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '1' };
USE jarvis_test;
