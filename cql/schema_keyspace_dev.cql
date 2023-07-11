CREATE KEYSPACE IF NOT EXISTS jarvis
WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '1' };
USE jarvis;