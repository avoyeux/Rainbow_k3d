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
        To get the data, only the instance attribute 'all_sdo_dates' needs to be called.
        """
        
        # CONSTANTs
        self._first_datetime: datetime = datetime(2012, 7, 23, 0, 6, 43)

        # RUN
        all_metadata = self._fetch_all_dates()
        used_datetimes = self._protuberance_datetimes()
        self._all_sdo_metadata = self._filter_dates(
            all_data=all_metadata,
            used_datetimes=used_datetimes,
        )

    @property
    def all_sdo_dates(self) -> list[SdoData]:
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

        # DATE range
        date_begin = datetime(2012, 7, 23, 0, 0, 0)
        date_end = datetime(2012, 7, 25, 12, 0, 0)  # ! check exactly what it should be

        # SDO client and search
        aia_client = SdoClientMedoc()
        aia_data_list = aia_client.search(
            dates=[date_begin, date_end],
            waves=['304'],
            series='aia.lev1',
            cadence=['12s'],
        )
        return aia_data_list

    def _protuberance_datetimes(self) -> list[datetime]:
        """
        To get the datetimes of the protuberance data.
        The datetimes are stored in the SDO_timestamps.txt file (path gotten from the config file).

        Returns:
            list[datetime]: the datetimes of used in the protuberance data.
        """

        # ! need to check for the exception cases if it still applies to the fetched data

        # TIMESTAMPs get
        with open(config.path.data.sdo_timestamp, 'r') as files:
            strings = files.read().splitlines()
        dates_list = [s.split(" ; ")[0][:-3] for s in strings]

        # DATETIMEs conversion
        dates_datetime = [datetime.strptime(s, '%Y-%m-%dT%H:%M:%S') for s in dates_list]
        return dates_datetime
    
    @Decorators.running_time
    def _filter_dates(
            self,
            all_data: list[SdoData],
            used_datetimes: list[datetime],
        ) -> list[SdoData]:
        """
        To filter the SDO data to keep only the ones that are needed.
        The needed datetimes are the ones used in the protuberance data and the ones with a time
        interval of 1 minute.

        Args:
            all_data (list[SdoData]): the SDO data to filter.
            used_datetimes (list[datetime]): the datetimes used in the protuberance data.

        Returns:
            list[SdoData]: the filtered SDO data.
        """

        # INDEXEs to keep
        used_indexes = [  # ? do I need to change this to add a break inside the loop ?
            i
            for i, data in enumerate(all_data)
            for used_date in used_datetimes
            if self._compare_datetimes_by_seconds(
                date1=cast(datetime, data.date_obs),
                date2=used_date,
            )
        ]
        new_indexes = [
            i
            for i, data in enumerate(all_data)
            if self._compare_datetimes_by_seconds(
                date1=cast(datetime, data.date_obs),
                date2=self._needed_datetime(i),
            )
        ]
        keep_indexes = sorted(list(set(used_indexes + new_indexes)))

        # SDO data filtering
        return [all_data[i] for i in keep_indexes]

    def _compare_datetimes_by_seconds(self, date1: datetime, date2: datetime) -> bool:
        """
        To compare the equality of two datetimes with a precision of seconds.
        If they are equal, it returns True, else False.

        Args:
            date1 (datetime): the first datetime to compare.
            date2 (datetime): the second datetime to compare.

        Returns:
            bool: True if the datetimes are equal, else False.
        """

        # FILTERs
        year_filter = (date1.year == date2.year)
        month_filter = (date1.month == date2.month)
        day_filter = (date1.day == date2.day)
        hour_filter = (date1.hour == date2.hour)
        minute_filter = (date1.minute == date2.minute)
        second_filter = (date1.second == date2.second)

        # COMPARISON
        comparison = all([
            year_filter,
            month_filter,
            day_filter,
            hour_filter,
            minute_filter,
            second_filter,
        ])
        return comparison

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
