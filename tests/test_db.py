from sqlalchemy import select

from batterygemma.db.models import QA, Chunk, Document
from batterygemma.db.session import get_session


def test_document_chunk_and_qa_roundtrip(engine):
    with get_session(engine) as s:
        doc = Document(
            doc_id="chemrxiv:abc", source="chemrxiv", external_id="abc", title="NMC811 cracking",
            norm_title="nmc811 cracking", license="CC-BY-4.0", topic_tags=["cathode", "NMC811"],
        )
        doc.chunks.append(
            Chunk(chunk_id="chemrxiv:abc#s3.2-c04", order=4, tokens=612, section_path=["3", "3.2"],
                  text="Above 4.2 V vs Li/Li+, NMC811 undergoes the H2-H3 phase transition.")
        )
        s.add(doc)
        s.add(
            QA(id="qa-1", doc_ids=["chemrxiv:abc"], chunk_ids=["chemrxiv:abc#s3.2-c04"], license="CC-BY-4.0",
               task_format="instruction_response", polarity="positive", question_type="mechanism",
               component="cathode", answer_type="OPEN",
               turns=[{"role": "user", "content": "Why?"}, {"role": "assistant", "content": "Because."}])
        )

    with get_session(engine) as s:
        loaded = s.scalars(select(Document)).one()
        assert loaded.topic_tags == ["cathode", "NMC811"]
        assert loaded.chunks[0].section_path == ["3", "3.2"]
        qa = s.get(QA, "qa-1")
        assert qa.turns[1]["role"] == "assistant"
        assert qa.status == "generated" and qa.tier == "silver"
