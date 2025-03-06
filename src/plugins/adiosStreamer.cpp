#include "adiosStreamer.hpp"

// Initialize the ADIOS2 client
adios_client_t::adios_client_t(MPI_Comm& comm, nrs_t *nrs)
                               : _comm(comm)
{
    // Set nekrs object
    _nrs = nrs;

    // Set MPI comm, rank and size 
    //_comm = platform->comm.mpiComm;
    MPI_Comm_rank(_comm, &_rank);
    MPI_Comm_size(_comm, &_size);

    // Set up adios2 parameters
    platform->options.getArgs("ADIOS ML ENGINE",_engine);
    platform->options.getArgs("ADIOS ML TRANSPORT",_transport);
    platform->options.getArgs("ADIOS ML STREAM",_stream);

    // Initialize adios2
    if (_rank == 0)
        printf("\nInitializing ADIOS2 client ...\n");
    try
    {
        _adios = new adios2::ADIOS(_comm);
        _io = _adios->DeclareIO("nekRS-ML");
        _io.SetEngine(_engine);
        if (_stream == "sync") {
            _params["RendezvousReaderCount"] = "1";
            _params["QueueFullPolicy"] = "Block";
            _params["QueueLimit"] = "1";
        } else if (_stream == "async") {
            _params["RendezvousReaderCount"] = "0";
            _params["QueueFullPolicy"] = "Block";
            _params["QueueLimit"] = "3";
        }
        _params["DataTransport"] = _transport;
        _params["OpenTimeoutSecs"] = "600";
        _io.SetParameters(_params);
    } 
    catch (std::exception &e)
    {
        printf("Exception, STOPPING PROGRAM from rank %d\n", _rank);
        std::cout << e.what() << "\n";
        fflush(stdout);
    }
    MPI_Barrier(_comm);
    if (_rank == 0)
        printf("All done\n");
    fflush(stdout);
}

// destructor
adios_client_t::~adios_client_t()
{
    if (_rank == 0) printf("Taking down adios_client_t\n");
}