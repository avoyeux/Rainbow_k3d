"""
To get and process the SDO dates.
The dates need to be of a cadence of 1 minute while some need to correspond exactly to the SDO data
used for the creation of the volumetric protuberance data.
Hence, all the SDO files dates need to be fetched (and the corresponding server filepaths) to then
choose the ones that have a cadence of 1 minute relative to the SDO data that is already used.
"""

# IMPORTs sub
from datetime import datetime, timedelta
from sitools2 import SdoClientMedoc  # IAS package for the SDO data

# IMPORTs personal
from common import config, Decorators

# TYPE ANNOTATIONs
from typing import cast
from sitools2.clients.sdo_data import SdoData

# API public
__all__ = ['AllSDOMetadata']

# ? should I keep the datetimes with 30 seconds difference when using the first dates from the 
# ? protuberance data ?



class AllSDOMetadata:
    """
    Uses PySitools2 to get the SDO acquisitions metadata and filters them to only keep the ones
    that are needed for the final warped integration plot.
    """

    def __init__(self) -> None:
        """
        To get the SDO metadata needed for the final warped integration plot.
        To get the data, only the instance attribute 'sdo_metadata' needs to be called.
        """
        
        # CONSTANTs
        self._first_datetime: datetime = datetime(2012, 7, 23, 0, 0, 43)
        self._search_date_begin: datetime = datetime(2012, 7, 23, 0, 0, 42)
        self._search_date_end: datetime = datetime(2012, 7, 25, 12, 0, 0)

        # RUN
        all_metadata = self._fetch_all_dates()
        used_datetimes = self._protuberance_datetimes()
        self._all_sdo_metadata = self._filter_dates(
            all_metadata=all_metadata,
            used_datetimes=used_datetimes,
        )

    @property
    def sdo_metadata(self) -> list[SdoData]:
        """
        Attribute that returns all the needed SDO metadata.
        The date_obs and .ias_path are of importance for me.

        Returns:
            list[SdoData]: the SDO metadata of the dates of interest.
        """

        return self._all_sdo_metadata
    
    @Decorators.running_time
    def _fetch_all_dates(self) -> list[SdoData]:
        """
        To fetch all the SDO metadata (given the date interval of interest).

        Returns:
            list[SdoData]: the SDO metadata of the dates of interest.
        """

        # SDO client and search
        aia_client = SdoClientMedoc()
        aia_data_list = aia_client.search(
            dates=[self._search_date_begin, self._search_date_end],
            waves=['304'],
            series='aia.lev1',
            cadence=['12s'],
        )
        return aia_data_list  # ? is the list sorted by date ?

    def _protuberance_datetimes(self) -> list[datetime]:
        """
        To get the datetimes of the protuberance data.
        The datetimes are stored in the SDO_timestamps.txt file (path gotten from the config file).

        Returns:
            list[datetime]: the datetimes of used in the protuberance data.
        """

        # ! need to check for the exception dates if it still applies to the fetched data

        # TIMESTAMPs get
        with open(config.path.data.sdo_timestamp, 'r') as files:
            strings = files.read().splitlines()
        dates_list = [s.split(" ; ")[1][:-3] for s in strings]

        # DATETIMEs conversion
        dates_datetime = [datetime.strptime(s, '%Y-%m-%dT%H:%M:%S') for s in dates_list]
        return dates_datetime
    
    @Decorators.running_time
    def _filter_dates(
            self,
            all_metadata: list[SdoData],
            used_datetimes: list[datetime],
        ) -> list[SdoData]:
        """
        To filter the SDO data to keep only the ones that are needed.
        The needed datetimes are the ones used in the protuberance data and the ones with a time
        interval of 1 minute.

        Args:
            all_metadata (list[SdoData]): the SDO metadata to filter.
            used_datetimes (list[datetime]): the datetimes used in the protuberance data.

        Returns:
            list[SdoData]: the filtered SDO data.
        """

        # NEW RANGE of dates
        date_range = self._search_date_end - self._search_date_begin
        date_range = int(date_range.total_seconds() // 60) + 1

        # INDEXEs to keep
        used_indexes: list[int] = cast(list[int], [None] * len(used_datetimes))
        for i, used_date in enumerate(used_datetimes):
            for j, metadata in enumerate(all_metadata):
                if cast(datetime, metadata.date_obs).replace(microsecond=0) == used_date:
                    used_indexes[i] = j
                    break
        new_indexes: list[int] = []
        for i in range(date_range):
            for j, metadata in enumerate(all_metadata):
                if (
                    cast(datetime, metadata.date_obs).replace(microsecond=0) ==
                    self._needed_datetime(i)
                ):
                    new_indexes.append(j) # ! 20 dates missing
                    break
        keep_indexes = sorted(list(set(used_indexes + new_indexes)))

        # SDO data filtering
        return [all_metadata[i] for i in keep_indexes]

    def _needed_datetime(self, time_coef: int) -> datetime:
        """
        To get the different datetimes needed from the SDO data.

        Args:
            time_coef (int): the coefficient to multiply the time difference with the first date.

        Returns:
            datetime: the datetime needed from the SDO data.
        """

        return self._first_datetime + timedelta(minutes=1 * time_coef)

    def check_dates(self) -> bool:

        # * need to compare the result with the dates that are in the 'SDO_timestamps.txt' file to
        # * make sure that the dates coincide with the ones in the text file.
        pass
