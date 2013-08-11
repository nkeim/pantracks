import tables
from .base import TracksWriter

class BigTracksWriter(TracksWriter):
    """Write particle tracks in the "BigTracks" format (a single large HDF5 table,
    indexed using pyables)
    
    Constructor arguments:
    'expectedrows': Optimize table to eventually hold this many rows
    'compress': Use high-performance blosc compression in the file
    """
    takes_expectedrows = True # To signal that __init__ wants to know this
    def __init__(self, filename, expectedrows=1000, compress=False):
        self.filename = filename
        self.outfile = tables.openFile(self.filename, 'w')
        if compress:
            self.trackstable = self.outfile.createTable('/', 'bigtracks', TrackPoint,
                    expectedrows=expectedrows,
                    filters=tables.Filters(complevel=5, complib='blosc'))
        else:
            self.trackstable = self.outfile.createTable('/', 'bigtracks', TrackPoint,
                    expectedrows=expectedrows)
    def append(self, rows):
        """Append rows to the file.

        Should be a pandas DataFrame with columns 'frame', 'particle', 'x', 'y',
        'intensity', 'rg2'.
        """
        # This is where one would also down-convert to float32
        pass

class BigTracksCompressedWriter(BigTracksWriter):
    """Version of BigTracksWriter that compresses the file.
    Suitable for passing to runtrackpy.track.track2disk()
    """
    def __init__(self, *args, **kw):
        kwc = kw.copy()
        kwc['compress'] = True
        super(BigTracksCompressedWriter, self).__init__(*args, **kwc)

# Tracks file indexing
def create_tracksfile_indices(tracksfilename):
    """Create indices for the tracks data in the HDF5 file 'tracksfilename'.
    Indices are necessary to efficiently access the data.

    This is only necessary if the normal tracking process (with track2disk())
    did not finish successfully.
    """
    outfile = tables.openFile(tracksfilename, 'a')
    try:
        trtab = outfile.root.bigtracks
        _create_table_indices(trtab)
    finally:
        outfile.close()
def _create_table_indices(trackstable):
    """Create indices on the tracks PyTables table."""
    trackstable.cols.frame.createIndex()
    trackstable.cols.particle.createIndex()

# Format of the tracks data file
class TrackPoint(tables.IsDescription):
    """pytables format for tracks data"""
    frame = tables.Float32Col(pos=1)
    particle = tables.Float32Col(pos=2)
    x = tables.Float32Col(pos=3)
    y = tables.Float32Col(pos=4)
    intensity = tables.Float32Col(pos=5)
    rg2 = tables.Float32Col(pos=6)


