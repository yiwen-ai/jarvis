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

-- Content Embeddings:
-- did     BLOB,                # document id, 12 bytes, https://docs.rs/xid/latest/xid/
-- lang    ASCII,               # document language
-- ids     FROZEN<LIST<ASCII>>, # content's nodes ids list
-- content BLOB,                # a well processed and segmented content list for embedding in CBOR format
-- tokens  MAP<ASCII, INT>,     # tokens uåsed, example: {"ada2": 299}, ada2 is text-embedding-ada-002
-- ada2    LIST<FLOAT>          # embedding by text-embedding-ada-002, 1536 dimensions
CREATE TABLE IF NOT EXISTS jarvis.ces (
    did     BLOB,
    lang    ASCII,
    ids     FROZEN<LIST<ASCII>>,
    content BLOB,
    tokens  MAP<ASCII, INT>,
    ada2    LIST<FLOAT>,
    PRIMARY KEY (did, lang, ids)
) WITH caching = {'enabled': 'true'}
    AND comment = 'text embedding'
    AND compaction = {'class': 'SizeTieredCompactionStrategy'}
    AND compression = {'sstable_compression': 'LZ4Compressor'}
    AND default_time_to_live = 0;

-- Translations:
-- did     BLOB,             # document id, 12 bytes, https://docs.rs/xid/latest/xid/
-- lang    ASCII,            # document language
-- ver     SMALLINT,         # document version by language
-- content BLOB,             # a well processed content list, from client input or default GPT model, gpt3.
-- tokens  MAP<ASCII, INT>,  # tokens uåsed, example: {"gpt3": 1299}, gpt3 is gpt-3.5-turbo
-- gpt4    BLOB              # optional content list, from GPT-4 model, gpt4.
CREATE TABLE IF NOT EXISTS jarvis.trs (
    did     BLOB,
    lang    ASCII,
    ver     SMALLINT,
    content BLOB,
    tokens  MAP<ASCII, INT>,
    gpt4    BlOB,
    PRIMARY KEY (did, lang, ver)
) WITH caching = {'enabled': 'true'}
    AND comment = 'text embedding'
    AND compaction = {'class': 'SizeTieredCompactionStrategy'}
    AND compression = {'sstable_compression': 'LZ4Compressor'}
    AND default_time_to_live = 0;