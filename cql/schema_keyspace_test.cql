CREATE KEYSPACE IF NOT EXISTS jarvis_test
WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '1' };

CREATE TABLE IF NOT EXISTS jarvis_test.counter
(
    uid                BLOB,
    embedding          COUNTER,
    embedding_tokens   COUNTER,
    translating        COUNTER,
    translating_tokens COUNTER,
    PRIMARY KEY (uid)
) WITH caching = {'enabled': 'true'}
    AND comment = 'user counters'
    AND compaction = {'class': 'SizeTieredCompactionStrategy'}
    AND compression = {'sstable_compression': 'LZ4Compressor'}
    AND default_time_to_live = 0;

CREATE TABLE IF NOT EXISTS jarvis_test.translating (
    did     BLOB,
    ver     SMALLINT,
    lang    ASCII,
    user    ASCII,
    tokens  MAP<ASCII, INT>,
    content BLOB,
    gpt4    BlOB,
    PRIMARY KEY (did, ver, lang)
) WITH CLUSTERING ORDER BY (ver DESC)
    AND caching = {'enabled': 'true'}
    AND comment = 'document translating'
    AND compaction = {'class': 'SizeTieredCompactionStrategy'}
    AND compression = {'sstable_compression': 'LZ4Compressor'}
    AND default_time_to_live = 0;

CREATE TABLE IF NOT EXISTS jarvis_test.embedding (
    did     BLOB,
    crc     BLOB,
    lang    ASCII,
    user    ASCII,
    tokens  MAP<ASCII, INT>,
    ids     ASCII,
    content BLOB,
    ada2    LIST<FLOAT>,
    PRIMARY KEY (did, crc)
) WITH caching = {'enabled': 'true'}
    AND comment = 'content embedding'
    AND compaction = {'class': 'SizeTieredCompactionStrategy'}
    AND compression = {'sstable_compression': 'LZ4Compressor'}
    AND default_time_to_live = 0;
