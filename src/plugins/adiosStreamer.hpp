#if !defined(nekrs_adiosstreamer_hpp_)
#define nekrs_adiosstreamer_hpp_

#include "nrs.hpp"
#include "adios2.h"

class adios_client_t
{
public:
  adios_client_t(MPI_Comm& comm, nrs_t *nrs);
  ~adios_client_t();

  // adios objects
  adios2::ADIOS *_adios;
  adios2::IO _io;

private:
  // Streamer parameters
  std::string _engine;
  std::string _transport;
  std::string _stream;

  // adios objects
  adios2::Params _params;

  // nekrs objects 
  nrs_t *_nrs;

  // MPI stuff
  int _rank, _size;
  MPI_Comm& _comm;

};

#endif