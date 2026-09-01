import os
import re
import zipfile
import logging.handlers


class ZippedTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """Rotates at midnight and zips the day that just ended.

    Uses the base class doRollover: it names the file after the date the log
    *covers* (from rolloverAt - interval, not now()) and honours backupCount.
    We only supply the two stdlib hooks: namer and rotator.

    backupCount counts zip files, one per day with when="midnight", so the
    number reads as days of history.
    """

    def __init__(self, filename, **kwargs):
        # The folder must exist before the base class opens the file.
        folder = os.path.dirname(os.path.abspath(filename))
        if folder:
            os.makedirs(folder, exist_ok=True)
        super().__init__(filename, **kwargs)
        self.namer = self._name
        self.rotator = self._compress

    @staticmethod
    def _name(default_name):
        """"app.log.2026-08-19" -> "app.log.2026-08-19.zip".

        A sequence number is appended if the name already exists. Without it
        the base class would delete the existing zip first, so a second
        rollover on the same date, after downtime across midnight, or because
        both master and worker rotate, ate a whole day of history.
        """
        candidate = f"{default_name}.zip"
        n = 2
        while os.path.exists(candidate):
            candidate = f"{default_name}.{n}.zip"
            n += 1
        return candidate

    @staticmethod
    def _compress(source, mal):
        with zipfile.ZipFile(mal, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(source, arcname=os.path.basename(source))
        os.remove(source)

    def getFilesToDelete(self):
        """Zip files falling outside backupCount, oldest first.

        Reimplemented because the base class extMatch varies across python
        versions in whether it accepts an extra ".zip" suffix. Without a match
        it deletes nothing and retention becomes infinite, which is the bug we had.
        """
        if self.backupCount <= 0:
            return []

        folder, base_name = os.path.split(self.baseFilename)
        monster = re.compile(
            rf"^{re.escape(base_name)}\.(\d{{4}}-\d{{2}}-\d{{2}}[\d\-_:.]*?)(?:\.(\d+))?\.zip$"
        )

        hit = []
        for name in os.listdir(folder or "."):
            m = monster.match(name)
            if m:
                # Sort on (period, sequence) so the oldest goes first.
                hit.append(((m.group(1), int(m.group(2) or 0)), os.path.join(folder, name)))

        hit.sort()
        if len(hit) <= self.backupCount:
            return []
        return [path for _, path in hit[: len(hit) - self.backupCount]]
