import os
import re
import zipfile
import logging.handlers


class ZippedTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """Roterer ved døgnskiftet og zipper døgnet som ble avsluttet.

    Bruker base-klassens doRollover, som gjør to ting riktig vi tidligere
    gjorde feil: den navngir filen etter datoen loggen *dekker* (utledet
    fra rolloverAt - interval, ikke fra now()), og den håndterer
    backupCount. Vi bidrar bare med de to hookene stdlib er laget for:
    namer (filnavnet) og rotator (komprimeringen).

    backupCount teller zip-filer. Med when="midnight" er det én per døgn,
    så tallet leses som antall dager historikk.
    """

    def __init__(self, filename, **kwargs):
        # Mappen må finnes før base-klassen åpner filen.
        mappe = os.path.dirname(os.path.abspath(filename))
        if mappe:
            os.makedirs(mappe, exist_ok=True)
        super().__init__(filename, **kwargs)
        self.namer = self._navngi
        self.rotator = self._komprimer

    @staticmethod
    def _navngi(standardnavn):
        """«app.log.2026-08-19» -> «app.log.2026-08-19.zip».

        Finnes navnet allerede, legges det på et løpenummer. Uten det
        ville base-klassen slette den eksisterende zipen først, og et
        ekstra rollover på samme dato — etter nedetid over døgnskiftet,
        eller fordi både master og worker roterer — spiste et helt døgn
        med historikk.
        """
        kandidat = f"{standardnavn}.zip"
        n = 2
        while os.path.exists(kandidat):
            kandidat = f"{standardnavn}.{n}.zip"
            n += 1
        return kandidat

    @staticmethod
    def _komprimer(kilde, mal):
        with zipfile.ZipFile(mal, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(kilde, arcname=os.path.basename(kilde))
        os.remove(kilde)

    def getFilesToDelete(self):
        """Zip-filene som faller utenfor backupCount, eldste først.

        Egen implementasjon fordi base-klassens extMatch varierer mellom
        python-versjoner i om den godtar et ekstra «.zip»-suffiks. Uten
        treff sletter den ingenting, og retensjonen blir uendelig — som
        var feilen her før.
        """
        if self.backupCount <= 0:
            return []

        mappe, grunnavn = os.path.split(self.baseFilename)
        monster = re.compile(
            rf"^{re.escape(grunnavn)}\.(\d{{4}}-\d{{2}}-\d{{2}}[\d\-_:.]*?)(?:\.(\d+))?\.zip$"
        )

        treff = []
        for navn in os.listdir(mappe or "."):
            m = monster.match(navn)
            if m:
                # Sorter på (periode, løpenummer) så eldste ryker først.
                treff.append(((m.group(1), int(m.group(2) or 0)), os.path.join(mappe, navn)))

        treff.sort()
        if len(treff) <= self.backupCount:
            return []
        return [sti for _, sti in treff[: len(treff) - self.backupCount]]
