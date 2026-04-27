import os
import zipfile
from datetime import datetime
import logging.handlers


class ZippedTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    def doRollover(self):
        print("they see me rollin'")
        """
        Overridden to zip the log file at the end of the day and start a new log file.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        # Prepare the name for the zipped log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        zipped_log_filename = f"{self.baseFilename}.{date_str}.zip"

        # Zip the current log file
        with zipfile.ZipFile(zipped_log_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(self.baseFilename, arcname=os.path.basename(self.baseFilename))

        # Remove the original log file
        os.remove(self.baseFilename)

        # Reset the rollover time and stream
        self.rolloverAt = self.rolloverAt + self.interval
        self.mode = 'w'
        self.stream = self._open()

        print("they hatin'")
