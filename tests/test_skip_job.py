"""The skip job against a fake database, API and model: job/skip_job.py.

requests.get/post/put are replaced, so nothing leaves the machine.
"""
import json
import logging

import pytest
import requests

import skip_job

DB = "http://db"
API = "http://api/"
MODEL = "http://model"
PDF = b"%PDF-1.7 test"


def response(status=200, body=b"", content_type="application/json"):
    r = requests.Response()
    r.status_code = status
    r._content = body if isinstance(body, bytes) else json.dumps(body).encode()
    r.headers["Content-Type"] = content_type
    r.encoding = "utf-8"
    return r


def document(nr, **extra):
    doc = {
        "dokumentIdent": {"dokumentaar": 2001, "dokumentnummer": nr,
                          "embetenummer": 200},
        "erElektroniskTinglyst": False,
        "rettsstiftelsestyper": ["TEST_KODE"],
        "filRevisjonId": f"rev{nr}",
    }
    doc.update(extra)
    return doc


def model_result(boxes=None, pages=None):
    if boxes is None:
        boxes = [{"page": 1, "x": 10, "y": 20, "width": 30, "height": 12,
                  "kilde": "begge", "yolo_conf": 0.8}]
    return {"boxes": boxes, "pipeline_version": "test-1",
            "pages": pages if pages is not None else [{"page": 1, "rotation": 0}]}


class Backend:
    """The three services the job talks to, keyed by document."""

    def __init__(self, documents):
        self.list_response = response(body=documents)
        self.status = {}          # docid -> response; default KLAR
        self.pdf = {}             # docid -> response; default a PDF
        self.model = {}           # filrevisjonid -> response or exception
        self.pdf_requests = []
        self.model_requests = []
        self.labels = {}          # docid -> (params, body)
        self.side_funn = {}       # filrevisjonid -> body

    def get(self, url, **kwargs):
        if url == f"{DB}/ubehandlede_dokumenter":
            return self.list_response
        if url.startswith(f"{DB}/dokumentstatus/"):
            docid = "_".join(url.rsplit("/", 3)[1:])
            return self.status.get(docid, response(body="KLAR_FOR_BEHANDLING"))
        if url.startswith(f"{API}intern/pantebok/gjenpart/"):
            self.pdf_requests.append((url, kwargs))
            docid = url.rsplit("/", 1)[1].split("?")[0]
            return self.pdf.get(docid, response(body=PDF,
                                                content_type="application/pdf"))
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url, params=None, data=None, headers=None, timeout=None):
        assert url == f"{MODEL}/model"
        self.model_requests.append({"params": params, "data": data,
                                    "headers": headers, "timeout": timeout})
        result = self.model.get(params["filrevisjonid"],
                                response(body=model_result()))
        if isinstance(result, Exception):
            raise result
        return result

    def put(self, url, params=None, json=None):
        if url.startswith(f"{DB}/labels/"):
            docid = "_".join(url.rsplit("/", 3)[1:])
            self.labels[docid] = (params, json)
        elif url.startswith(f"{DB}/ml-side-funn/"):
            self.side_funn[url.rsplit("/", 1)[1]] = json
        else:
            raise AssertionError(f"unexpected PUT {url}")
        return response(body={})


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", DB)
    monkeypatch.setenv("API_URL", API)
    monkeypatch.setenv("MODEL_URL", MODEL)

    def install(documents):
        fake = Backend(documents)
        monkeypatch.setattr(requests, "get", fake.get)
        monkeypatch.setattr(requests, "post", fake.post)
        monkeypatch.setattr(requests, "put", fake.put)
        return fake

    return install


def test_happy_path(backend):
    fake = backend([document(1)])
    skip_job.hentDokumenterTilSladding()

    (url, kwargs), = fake.pdf_requests
    assert url == f"{API}intern/pantebok/gjenpart/2001_1_200?attestering=false"
    assert kwargs.get("timeout")

    call, = fake.model_requests
    assert call["params"] == {"elektronisk_tinglyst": "false",
                              "rettsstiftelsestyper": ["TEST_KODE"],
                              "filrevisjonid": "rev1"}
    assert call["data"] == PDF
    assert call["headers"]["Content-Type"] == "application/pdf"
    assert call["timeout"] == 600

    params, labels = fake.labels["2001_1_200"]
    assert params == {"mlFerdigBehandlet": "true", "mlPipelineVersjon": "test-1"}
    label, = labels
    assert isinstance(label.pop("id"), str)
    assert label == {
        "dokumentaar": 2001, "dokumentnummer": 1, "embetenummer": 200,
        "sidetall": 1, "type": "PERSONNUMMER",
        "height": 12, "width": 30, "x": 10, "y": 20,
        "mlGenerated": True, "mlStatus": "", "mlKilde": "begge",
        "mlConf": 0.8, "mlPipelineVersjon": "test-1",
    }


def test_elektronisk_tinglyst_is_passed_on(backend):
    fake = backend([document(1, erElektroniskTinglyst=True)])
    skip_job.hentDokumenterTilSladding()
    assert fake.model_requests[0]["params"]["elektronisk_tinglyst"] == "true"


def test_rotated_pages_become_side_funn(backend):
    fake = backend([document(1)])
    fake.model["rev1"] = response(body=model_result(pages=[
        {"page": 1, "rotation": 0}, {"page": 2, "rotation": 90}]))
    skip_job.hentDokumenterTilSladding()
    assert fake.side_funn["rev1"] == [
        {"sidetall": 2, "funnType": "ROTERT_SIDE", "verdi": "90"}]


def test_side_funn_is_cleared_when_no_page_is_rotated(backend):
    fake = backend([document(1)])
    skip_job.hentDokumenterTilSladding()
    assert fake.side_funn["rev1"] == []


def test_no_boxes_still_marks_the_document_processed(backend):
    fake = backend([document(1)])
    fake.model["rev1"] = response(body=model_result(boxes=[]))
    skip_job.hentDokumenterTilSladding()
    params, labels = fake.labels["2001_1_200"]
    assert labels == []
    assert params["mlFerdigBehandlet"] == "true"


@pytest.mark.parametrize("status", [
    response(body="UNDER_BEHANDLING"),
    response(status=500, body={}),
])
def test_document_not_ready_is_skipped(backend, status):
    fake = backend([document(1), document(2)])
    fake.status["2001_1_200"] = status
    skip_job.hentDokumenterTilSladding()
    assert [url for url, _ in fake.pdf_requests] == [
        f"{API}intern/pantebok/gjenpart/2001_2_200?attestering=false"]
    assert list(fake.labels) == ["2001_2_200"]


@pytest.mark.parametrize("pdf", [
    response(status=401, body={"error": "Dokumentet er skjermet"}),
    response(status=500, body=b"error", content_type="text/plain"),
    response(body=b"<html>login</html>", content_type="text/html"),
], ids=["skjermet", "server-error", "not-a-pdf"])
def test_pdf_that_cannot_be_fetched_is_skipped(backend, pdf):
    fake = backend([document(1), document(2)])
    fake.pdf["2001_1_200"] = pdf
    skip_job.hentDokumenterTilSladding()
    assert [c["params"]["filrevisjonid"] for c in fake.model_requests] == ["rev2"]
    assert list(fake.labels) == ["2001_2_200"]
    assert list(fake.side_funn) == ["rev2"]


@pytest.mark.parametrize("failure", [
    response(status=500, body=b"CUDA out of memory", content_type="text/plain"),
    requests.exceptions.Timeout(),
    requests.exceptions.ConnectionError(),
], ids=["error-status", "timeout", "connection"])
def test_model_failure_leaves_the_document_unprocessed(backend, failure):
    # No PUT, so the document stays in ubehandlede_dokumenter for the next run.
    fake = backend([document(1), document(2)])
    fake.model["rev1"] = failure
    skip_job.hentDokumenterTilSladding()
    assert list(fake.labels) == ["2001_2_200"]
    assert list(fake.side_funn) == ["rev2"]


def test_failing_document_list_raises(backend):
    fake = backend([])
    fake.list_response = response(status=503, body={})
    with pytest.raises(Exception, match="unprocessed documents"):
        skip_job.hentDokumenterTilSladding()


@pytest.mark.xfail(strict=True, raises=AttributeError,
                   reason="a document without dokumentIdent stops the whole run")
def test_document_without_ident_does_not_stop_the_run(backend):
    fake = backend([{"filRevisjonId": "broken"}, document(2)])
    skip_job.hentDokumenterTilSladding()
    assert list(fake.labels) == ["2001_2_200"]


def test_log_lines_are_json():
    record = logging.LogRecord("skip_job", logging.ERROR, __file__, 1,
                               'Status "%s" for Ørsta', ("KLAR",), None)
    line = skip_job.JsonFormatter().format(record)
    assert "\n" not in line
    parsed = json.loads(line)
    assert parsed["level"] == "ERROR"
    assert parsed["message"] == 'Status "KLAR" for Ørsta'
    assert "Ø" in line
