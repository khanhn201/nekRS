#if !defined(nekrs_trajGen_hpp_)
#define nekrs_trajGen_hpp_

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include <filesystem>

void deleteDirectoryContents(const std::filesystem::path& dir);


class trajGen_t 
{
public:
    trajGen_t(nrs_t *nrs, int dt_factor_, dfloat time_init_);
    ~trajGen_t(); 

    // public variables
    std::string writePath;
    dfloat time_init;
    int dt_factor;
    dfloat *previous_step;
    
    // member functions 
    void trajGenSetup();
    void trajGenWrite(dfloat time, int tstep, const std::string& field_name);

private:
    // nekrs objects 
    nrs_t *nrs;
    mesh_t *mesh;
    ogs_t *ogs;

    // MPI stuff 
    int rank;
    int size;

    // for prints 
    bool verbose = true; 

    // for writing 
    bool write = true;
};

#endif
