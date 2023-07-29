CREATE TABLE IF NOT EXISTS translating (
    gid      BLOB,     -- group id, content belong to
    cid      BLOB,     -- creation id, 12 bytes, https://docs.rs/xid/latest/xid/
    language TEXT,     -- content's language, ISO 639-3
    version  SMALLINT, -- creation version
    model    TEXT,     -- model used for translating, example: "gpt3.5"
    tokens   INT,      -- tokens uåsed, example: {"gpt3.5": 1299}
    content  BLOB,     -- a well pruned content in CBOR format
    error    TEXT,     -- error message
    PRIMARY KEY (gid, cid, language, version)
) WITH CLUSTERING ORDER BY (cid DESC, language ASC, version DESC)
    AND caching = {'enabled': 'true'}
    AND comment = 'content translating'
    AND compaction = {'class': 'SizeTieredCompactionStrategy'}
    AND compression = {'sstable_compression': 'LZ4Compressor'}
    AND default_time_to_live = 0;

CREATE TABLE IF NOT EXISTS summarizing (
    gid      BLOB,     -- group id, content belong to
    cid      BLOB,     -- creation id, 12 bytes, https://docs.rs/xid/latest/xid/
    language TEXT,     -- content's language, ISO 639-3
    version  SMALLINT, -- creation version
    model    TEXT,     -- model used for translating, example: "gpt3.5"
    tokens   INT,      -- tokens uåsed, example: {"gpt3.5": 1299}
    summary  TEXT,     -- summary
    error    TEXT,    -- error message
    PRIMARY KEY (gid, cid, language, version)
) WITH CLUSTERING ORDER BY (cid DESC, language ASC, version DESC)
    AND caching = {'enabled': 'true'}
    AND comment = 'content summarizing'
    AND compaction = {'class': 'SizeTieredCompactionStrategy'}
    AND compression = {'sstable_compression': 'LZ4Compressor'}
    AND default_time_to_live = 0;

CREATE TABLE IF NOT EXISTS embedding (
    uuid     BLOB, -- 16 bytes, SHA3-256(cid+lang+ids)[..16], used for qdrant
    cid      BLOB, -- creation id, 12 bytes, https://docs.rs/xid/latest/xid/
    language TEXT, -- content's language, ISO 639-3
    version  SMALLINT, -- creation version
    ids      TEXT, -- content's nodes ids list joined by ','
    gid      BLOB, -- group id, content belong to
    content  BLOB, -- a well processed and segmented content for embedding in CBOR format
    PRIMARY KEY (uuid)
) WITH caching = {'enabled': 'true'}
    AND comment = 'content embedding'
    AND compaction = {'class': 'SizeTieredCompactionStrategy'}
    AND compression = {'sstable_compression': 'LZ4Compressor'}
    AND default_time_to_live = 0;

CREATE INDEX embedding_gid ON embedding (cid);
