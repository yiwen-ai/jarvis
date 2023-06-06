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
-- lang    ASCII,            # document language
-- ver     SMALLINT,         # document version by language
-- user    ASCII,
-- tokens  MAP<ASCII, INT>,  # tokens uåsed, example: {"gpt3.5": 1299}, gpt3 is gpt-3.5-turbo
-- content BLOB,             # a well processed content list, from client input or default GPT model, gpt3.5.
-- gpt4    BLOB              # optional content list, from GPT-4 model, gpt4.
CREATE TABLE IF NOT EXISTS jarvis.translating (
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

-- did     BLOB,                # document id, 12 bytes, https://docs.rs/xid/latest/xid/
-- lang    ASCII,               # document language
-- ids     FROZEN<LIST<ASCII>>, # content's nodes ids list
-- user    ASCII,
-- tokens  MAP<ASCII, INT>,     # tokens uåsed, example: {"ada2": 299}, ada2 is text-embedding-ada-002
-- content BLOB,                # a well processed and segmented content list for embedding in CBOR format
-- ada2    LIST<FLOAT>          # embedding by text-embedding-ada-002, 1536 dimensions
CREATE TABLE IF NOT EXISTS jarvis.embedding (
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
