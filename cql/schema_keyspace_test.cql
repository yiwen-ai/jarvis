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
    lang    ASCII,
    ver     SMALLINT,
    user    ASCII,
    tokens  MAP<ASCII, INT>,
    content BLOB,
    gpt4    BlOB,
    PRIMARY KEY (did, lang, ver)
) WITH caching = {'enabled': 'true'}
    AND comment = 'document translating'
    AND compaction = {'class': 'SizeTieredCompactionStrategy'}
    AND compression = {'sstable_compression': 'LZ4Compressor'}
    AND default_time_to_live = 0;

CREATE TABLE IF NOT EXISTS jarvis_test.embedding (
    did     BLOB,
    lang    ASCII,
    ids     FROZEN<LIST<ASCII>>,
    user    ASCII,
    tokens  MAP<ASCII, INT>,
    content BLOB,
    ada2    LIST<FLOAT>,
    PRIMARY KEY (did, lang, ids)
) WITH caching = {'enabled': 'true'}
    AND comment = 'content embedding'
    AND compaction = {'class': 'SizeTieredCompactionStrategy'}
    AND compression = {'sstable_compression': 'LZ4Compressor'}
    AND default_time_to_live = 0;
