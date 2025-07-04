#!/bin/bash
set -e -a

: ${NRSCONFIG_NOBUILD:=0}

if [ -z ${CC} ] || [ -z ${CXX}  ] || [ -z ${FC}  ]; then
  echo -e "\033[31mWARNING: env-var CC, CXX or FC is not set!\033[m"
  export CC=mpicc
  export CXX=mpic++
  export FC=mpif77
  read -p         "Press ENTER to continue with CC=$CC CXX=$CXX FC=$FC or ctrl-c to cancel"
fi

cmd="cmake -S . -B build -Wfatal-errors $@"
echo $cmd
eval $cmd
if [ $? -eq 0 ] && [ ${NRSCONFIG_NOBUILD} -eq 0 ]; then
  cmd="cmake --build ./build --target install -j8"
  echo ""
  echo $cmd
  echo -e "\033[32mPlease check the summary above carefully and press ENTER to continue or ctrl-c to cancel\033[m"
  read -rsn1 key

  eval $cmd
  if [ $? -eq 0 ]; then
    echo ""
    echo -e "\033[35mHooray! You're all set. The installation is complete.\033[m"
    echo ""
  fi
fi
