CREATE KEYSPACE IF NOT EXISTS jarvis
WITH replication = { 'class': 'NetworkTopologyStrategy', 'replication_factor': '3' };
USE jarvis;