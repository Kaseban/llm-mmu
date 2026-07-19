from mmu.proxy.sse import SSEParser


def test_single_event():
    p = SSEParser()
    evs = p.feed(b'event: message_start\ndata: {"type":"message_start"}\n\n')
    assert len(evs) == 1
    assert evs[0].event == "message_start"
    assert evs[0].json() == {"type": "message_start"}


def test_split_across_chunks():
    p = SSEParser()
    assert p.feed(b"event: message_de") == []
    assert p.feed(b"lta\ndata: {\"a\"") == []
    evs = p.feed(b": 1}\n\n")
    assert len(evs) == 1
    assert evs[0].event == "message_delta"
    assert evs[0].json() == {"a": 1}


def test_multiline_data_and_comments():
    p = SSEParser()
    evs = p.feed(b": keepalive\ndata: line1\ndata: line2\n\n")
    assert len(evs) == 1
    assert evs[0].data == "line1\nline2"


def test_crlf_and_garbage_tolerated():
    p = SSEParser()
    evs = p.feed(b"data: {\"ok\":true}\r\n\r\n\xff\xfe\n")
    assert len(evs) == 1
    assert evs[0].json() == {"ok": True}


def test_openai_done_sentinel():
    p = SSEParser()
    evs = p.feed(b"data: [DONE]\n\n")
    assert len(evs) == 1
    assert evs[0].data == "[DONE]"
    assert evs[0].json() is None
