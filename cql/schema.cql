CREATE TABLE IF NOT EXISTS jarvis.cnt
(
    key TEXT,
    cnt COUNTER,
    PRIMARY KEY (key)
) WITH caching = {'enabled': 'true'}
    AND comment = 'object counters'
    AND compaction = {'class': 'SizeTieredCompactionStrategy'}
    AND compression = {'sstable_compression': 'LZ4Compressor'}
    AND default_time_to_live = 0;

-- did  BLOB,       # document id
-- ids  TEXT,       # txt's nodes id, join by ','
-- lang TEXT,       # document language
-- txt  TEXT,       # wellformed node text of document for embedding
-- ada2 LIST<FLOAT> # embedding by text-embedding-ada-002, 1536 dimensions
CREATE TABLE IF NOT EXISTS jarvis.te (
    did  BLOB,
    ids  TEXT,
    lang TEXT,
    txt  TEXT,
    ada2 LIST<FLOAT>,
    PRIMARY KEY (did, ids, lang)
) WITH caching = {'enabled': 'true'}
    AND comment = 'text embedding'
    AND compaction = {'class': 'SizeTieredCompactionStrategy'}
    AND compression = {'sstable_compression': 'LZ4Compressor'}
    AND default_time_to_live = 0;
