//Hammad Amer 22i-0877
//Shayaan Khalid 22i-0863
//CS-J

//MPI FILE

#include <bits/stdc++.h>
#include <mpi.h>
#include <metis.h>
using namespace std;
using Edge = pair<int, double>;
const double INF = numeric_limits<double>::infinity();

// check MPI errors 
inline void MPI_Assert(int err, const char* msg) {
    if (err != MPI_SUCCESS) {
        char errstr[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(err, errstr, &len);
        fprintf(stderr, "MPI Error: %s : %s\n", msg, errstr);
        MPI_Abort(MPI_COMM_WORLD, err);
    }
}

// MAIN PARTITRION OF GRAPH
vector<int> metisPartition(const vector<vector<Edge>> &G, int nparts) {
    int n = G.size();
    vector<idx_t> xadj(n+1), adjncy;
    vector<idx_t> adjwgt;
    idx_t edgeCount = 0;
    for (auto &row : G) edgeCount += row.size();
    adjncy.reserve(edgeCount);
    adjwgt.reserve(edgeCount);
    xadj[0] = 0;
    for (int i = 0; i < n; ++i) {
        for (auto &e : G[i]) {
            adjncy.push_back(e.first);
            adjwgt.push_back((idx_t)ceil(e.second));
        }
        xadj[i+1] = adjncy.size();
    }
    vector<int> part(n);
    idx_t objval, ncon = 1, np = nparts;
    int ret = METIS_PartGraphKway(&n, &ncon, xadj.data(), adjncy.data(), nullptr,
                                  adjwgt.data(), nullptr,
                                  &np, nullptr, nullptr, nullptr,
                                  &objval, (idx_t*)part.data());
    if (ret != METIS_OK) {
        fprintf(stderr, "METIS_PartGraphKway failed with error code %d\n", ret);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return part;
}

// CREATES AN ADJACENCY LIST
void readGraph(const string &file, int &n, vector<vector<Edge>> &G) {
    ifstream in(file);
    if (!in) MPI_Abort(MPI_COMM_WORLD, 1);
    int u, v; double w;
    int mx = -1;
    vector<tuple<int,int,double>> edges;
    while (in >> u >> v >> w) {
        if (u < 0 || v < 0) continue;
        mx = max(mx, max(u, v));
        edges.emplace_back(u, v, w);
    }
    n = mx + 1;
    G.assign(n, {});
    for (auto &t : edges) {
        tie(u, v, w) = t;
        G[u].emplace_back(v, w);
        G[v].emplace_back(u, w);
    }
}

// SSSP TREE READER
void readSSSP(const string &file, vector<double> &Dist, vector<int> &Parent, int n) {
    ifstream in(file);
    if (!in) MPI_Abort(MPI_COMM_WORLD, 1);
    Dist.assign(n, INF);
    Parent.assign(n, -1);
    int v, p; double d;
    while (in >> v >> d >> p) {
        if (v >= 0 && v < n) {
            Dist[v] = (d < 0 ? INF : d);
            Parent[v] = p;
        }
    }
    if (n > 0) { Dist[0] = 0.0; Parent[0] = -1; }
}

// UPDATES READER
struct Update { bool isInsert; int u, v; double w; };
vector<Update> readUpdates(const string &file) {
    ifstream in(file);
    if (!in) MPI_Abort(MPI_COMM_WORLD, 1);
    vector<Update> updates;
    string line;
    while (getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        istringstream ss(line);
        char typ; ss >> typ;
        if (typ == '-') {
            int u, v; ss >> u >> v;
            updates.push_back({false, u, v, 0.0});
        } else if (typ == '+') {
            int u, v; double w; ss >> u >> v >> w;
            updates.push_back({true, u, v, w});
        }
    }
    return updates;
}


bool removeEdge(vector<vector<Edge>> &G, int a, int b) {
    if (a < 0 || b < 0 || a >= (int)G.size() || b >= (int)G.size()) return false;
    auto &ga = G[a];
    ga.erase(remove_if(ga.begin(), ga.end(), [&](const Edge &e){ return e.first == b; }), ga.end());
    auto &gb = G[b];
    gb.erase(remove_if(gb.begin(), gb.end(), [&](const Edge &e){ return e.first == a; }), gb.end());
    return true;
}

int main(int argc, char** argv) {
    MPI_Assert(MPI_Init(&argc, &argv), "MPI_Init");
    int rank, size;
    MPI_Assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    MPI_Assert(MPI_Comm_size(MPI_COMM_WORLD, &size), "MPI_Comm_size");

    if (argc != 5 && rank == 0) {
        cerr << "Usage: " << argv[0] << " <graph> <sssp> <updates> <output>\n";
    }

    double t0 = MPI_Wtime();

    int n;
    vector<vector<Edge>> G;
    vector<int> part;
    if (rank == 0) {
        readGraph(argv[1], n, G);
        part = metisPartition(G, size);
    }
    MPI_Assert(MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast n");
    if (rank != 0) part.resize(n);
    MPI_Assert(MPI_Bcast(part.data(), n, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast part");

    vector<int> localVerts;
    for (int i = 0; i < n; ++i)
        if (part[i] == rank) localVerts.push_back(i);
    int ln = localVerts.size();
    unordered_map<int,int> g2l;
    for (int i = 0; i < ln; ++i) g2l[localVerts[i]] = i;
    vector<vector<Edge>> localG(ln);
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            vector<pair<int, vector<Edge>>> sendBuf;
            for (int gi : localVerts) {
                if (part[gi] == r) {
                    int li = g2l[gi];
                    vector<Edge> edges;
                    for (auto &e : G[gi]) if (part[e.first] == r)
                        edges.emplace_back(e.first, e.second);
                    sendBuf.emplace_back(gi, edges);
                }
            }
            if (r == 0) {
                for (auto &p : sendBuf) {
                    int gi = p.first, li = g2l[gi];
                    for (auto &e : p.second) localG[li].emplace_back(g2l[e.first], e.second);
                }
            } else {
                int count = sendBuf.size();
                MPI_Send(&count, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                for (auto &p : sendBuf) {
                    int gi = p.first;
                    int esz = p.second.size();
                    MPI_Send(&gi, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                    MPI_Send(&esz, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                    for (auto &e : p.second) {
                        int nb = e.first;
                        double w = e.second;
                        MPI_Send(&nb, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                        MPI_Send(&w, 1, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
                    }
                }
            }
        }
    } else {
        MPI_Status st;
        int count;
        MPI_Recv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
        for (int i = 0; i < count; ++i) {
            int gi, esz;
            MPI_Recv(&gi, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
            MPI_Recv(&esz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
            int li = g2l[gi];
            for (int j = 0; j < esz; ++j) {
                int nb; double w;
                MPI_Recv(&nb, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
                MPI_Recv(&w, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &st);
                localG[li].emplace_back(g2l[nb], w);
            }
        }
    }
    double t1 = MPI_Wtime();

    vector<double> Dist;
    vector<int> Parent;
    readSSSP(argv[2], Dist, Parent, n);
    vector<double> globalDist = Dist;
    MPI_Assert(MPI_Allreduce(Dist.data(), globalDist.data(), n, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD), "MPI_Allreduce");
    Dist.swap(globalDist);
    double t2 = MPI_Wtime();

    vector<Update> updates;
    if (rank == 0) updates = readUpdates(argv[3]);
    int ucount = updates.size();
    MPI_Assert(MPI_Bcast(&ucount, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast ucount");
    if (rank != 0) updates.resize(ucount);
    vector<int> buf(ucount*3);
    vector<double> wbuf(ucount);
    if (rank == 0) {
        for (int i = 0; i < ucount; ++i) {
            buf[3*i]   = updates[i].isInsert;
            buf[3*i+1] = updates[i].u;
            buf[3*i+2] = updates[i].v;
            wbuf[i]    = updates[i].w;
        }
    }
    MPI_Assert(MPI_Bcast(buf.data(), ucount*3, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast buf");
    MPI_Assert(MPI_Bcast(wbuf.data(), ucount, MPI_DOUBLE, 0, MPI_COMM_WORLD), "MPI_Bcast wbuf");
    if (rank != 0) {
        for (int i = 0; i < ucount; ++i) {
            updates[i].isInsert = buf[3*i];
            updates[i].u        = buf[3*i+1];
            updates[i].v        = buf[3*i+2];
            updates[i].w        = wbuf[i];
        }
    }
    double t3 = MPI_Wtime();
 // updates locally on localG[]
    vector<bool> A_del(n,false), A_ins(n,false);
    for (auto &u : updates) {
        if (u.u < n && u.v < n && part[u.u] == rank && part[u.v] == rank) {
            int lu = g2l[u.u], lv = g2l[u.v];
            if (!u.isInsert) {
                removeEdge(localG, lu, lv);
                if (Parent[u.v] == u.u || Parent[u.u] == u.v) {
                    int r = (Dist[u.u] > Dist[u.v] ? u.u : u.v);
                    Dist[r] = INF; Parent[r] = -1;
                    A_del[r] = A_ins[r] = true;
                }
            } else {
                double w = u.w;
                localG[lu].emplace_back(lv, w);
                localG[lv].emplace_back(lu, w);
                int a = (Dist[u.u] <= Dist[u.v] ? u.u : u.v);
                int b = (a == u.u ? u.v : u.u);
                double nd = Dist[a] + w;
                if (nd < Dist[b]) {
                    Dist[b]   = nd;
                    Parent[b] = a;
                    A_ins[b]  = true;
                }
            }
        }
    }
    double t4 = MPI_Wtime();

      // Propagate deletions 
    queue<int> q;
    for (int i = 0; i < n; ++i) if (A_del[i]) q.push(i);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        int c = Parent[u];
        if (c >= 0 && !A_del[c]) {
            A_del[c] = A_ins[c] = true;
            Dist[c] = INF; Parent[c] = -1;
            q.push(c);
        }
    }
    double t5 = MPI_Wtime();

    //relaxing our code
    bool changed;
    do {
        changed = false;
        for (int v = 0; v < n; ++v) {
            if (A_ins[v] || A_del[v]) {
                for (auto &e : localG[g2l[v]]) {
                    int u = e.first;
                    double w = e.second;
                    double nd = Dist[v] + w;
                    if (nd < Dist[u]) {
                        if (nd < Dist[u]) {
                            Dist[u]   = nd;
                            Parent[u] = v;
                            changed    = true;
                        }
                    }
                }
            }
        }
    } while (changed);
    double t6 = MPI_Wtime();

    vector<double> finalD(n);
    vector<int> finalP(n);
    MPI_Assert(MPI_Reduce(Dist.data(),  finalD.data(), n, MPI_DOUBLE, MPI_MIN,  0, MPI_COMM_WORLD), "MPI_Reduce dist");
    MPI_Assert(MPI_Reduce(Parent.data(), finalP.data(), n, MPI_INT,    MPI_MAX, 0, MPI_COMM_WORLD), "MPI_Reduce parent");
    double t7 = MPI_Wtime();

    if (rank == 0) {
        ofstream out(argv[4]);
        out << "#vertex distance parent\n";
        for (int i = 0; i < n; ++i) {
            out << i << " "
                << (finalD[i] == INF ? -1.0 : finalD[i])
                << " " << finalP[i] << "\n";
        }
        printf("Timings (s): Partition=%.3f SSSPsync=%.3f UpdRead=%.3f Apply=%.3f Prop=%.3f Relax=%.3f Reduce=%.3f Total=%.3f\n",
               t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t7-t0);
    }

    MPI_Finalize();
    return 0;
}
