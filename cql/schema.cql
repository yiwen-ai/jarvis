-- uid                BLOB,    # user id, 12 bytes, https://docs.rs/xid/latest/xid/
-- embedding          COUNTER,
-- embedding_tokens   COUNTER,
-- translating        COUNTER,
-- translating_tokens COUNTER,
CREATE TABLE IF NOT EXISTS jarvis.counter
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

-- did     BLOB,             # document id, 12 bytes, https://docs.rs/xid/latest/xid/
-- ver     SMALLINT,         # document version, <= 32767
-- lang    ASCII,            # document language
-- user    ASCII,
-- tokens  MAP<ASCII, INT>,  # tokens uåsed, example: {"gpt3.5": 1299}, gpt3 is gpt-3.5-turbo
-- content BLOB,             # a well processed content list, from client input or default GPT model, gpt3.5.
-- gpt4    BLOB              # optional content list, from GPT-4 model, gpt4.
CREATE TABLE IF NOT EXISTS jarvis.translating (
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

-- did     BLOB,            # document id, 12 bytes, https://docs.rs/xid/latest/xid/
-- crc     BLOB,            # 4 bytes, CRC32(lang:ids), uuid = did + crc
-- lang    ASCII,           # document language
-- user    ASCII,
-- tokens  MAP<ASCII, INT>, # tokens uåsed, example: {"ada2": 299}, ada2 is text-embedding-ada-002
-- ids     ASCII,           # content's nodes ids list joined by ','
-- content BLOB,            # a well processed and segmented content list for embedding in CBOR format
-- ada2    LIST<FLOAT>      # embedding by text-embedding-ada-002, 1536 dimensions
CREATE TABLE IF NOT EXISTS jarvis.embedding (
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
