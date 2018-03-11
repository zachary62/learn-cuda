/*
    We have a grid size of (29, 13, 1) and blocks of size (32, 32, 1). This
    amounts to 29*13*32*32=386,048 total threads. The image however, is of size
    (400, 900) for a total pixel amount of 360,000. Thus, we have 26,048 idle
    threads.
/*
