//Hammad Amer 22i-0877
//Shayaan Khalid 22i-0863
//CS-J

//SEQUENTIAL FILE

#include <bits/stdc++.h>
using namespace std;
using Edge = pair<int,double>;
const double INF = numeric_limits<double>::infinity();

struct Update {
    bool isInsert;
    int u, v;
    double w;
};

void readGraph(const string &f, int &n, vector<vector<Edge>> &G) {
    ifstream in(f);
    int u, v; double w;
    int mx = -1;
    vector<tuple<int,int,double>> E;
    while (in >> u >> v >> w) {
        E.emplace_back(u, v, w);
        mx = max({mx, u, v});
    }
    n = mx + 1;
    G.assign(n, {});
    for (auto &t : E) {
        tie(u, v, w) = t;
        G[u].emplace_back(v, w);
        G[v].emplace_back(u, w);
    }
}

void readSSSP(const string &f, vector<double> &Dist, vector<int> &Parent, int n) {
    ifstream in(f);
    Dist.assign(n, INF);
    Parent.assign(n, -1);
    int v, p; double d;
    while (in >> v >> d >> p) {
        if (v >= 0 && v < n) {
            Dist[v] = (d < 0 ? INF : d);
            Parent[v] = p;
        }
    }
    // Force source=0
    Dist[0] = 0;
    Parent[0] = -1;
}

vector<Update> readUpdates(const string &f) {
    ifstream in(f);
    vector<Update> U;
    string line;
    while (getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        istringstream ss(line);
        vector<string> tok;
        string t;
        while (ss >> t) tok.push_back(t);
        if (tok.size() == 2) {
            int u = stoi(tok[0]), v = stoi(tok[1]);
            U.push_back({false, u, v, 0.0});
        } else if (tok.size() == 3) {
            int u = stoi(tok[0]), v = stoi(tok[1]);
            double w = stod(tok[2]);
            U.push_back({true, u, v, w});
        }
    }
    return U;
}

void removeEdge(vector<vector<Edge>> &G, int a, int b) {
    auto &A = G[a];
    A.erase(remove_if(A.begin(), A.end(), [&](auto &e) { return e.first == b; }), A.end());
    auto &B = G[b];
    B.erase(remove_if(B.begin(), B.end(), [&](auto &e) { return e.first == a; }), B.end());
}

void applyDeletions(const vector<Update> &U,
    vector<double> &Dist, vector<int> &Parent,
    vector<bool> &A_del, vector<bool> &A_ins,
    vector<vector<int>> &children,
    vector<vector<Edge>> &G)
{
    for (auto &u : U) {
        if (!u.isInsert) {
            int x = u.u, y = u.v;
            removeEdge(G, x, y);
            if (Parent[y] == x || Parent[x] == y) {
                int r = (Dist[x] > Dist[y] ? x : y);
                Dist[r] = INF;
                Parent[r] = -1;
                A_del[r] = A_ins[r] = true;
            }
        }
    }
}

void applyInsertions(const vector<Update> &U,
    vector<double> &Dist, vector<int> &Parent,
    vector<bool> &A_ins,
    vector<vector<Edge>> &G)
{
    for (auto &u : U) if (u.isInsert) {
        int x = u.u, y = u.v; double w = u.w;
        G[x].emplace_back(y, w);
        G[y].emplace_back(x, w);
        int a = (Dist[x] <= Dist[y] ? x : y);
        int b = (a == x ? y : x);
        if (Dist[b] > Dist[a] + w) {
            Dist[b] = Dist[a] + w;
            Parent[b] = a;
            A_ins[b] = true;
        }
    }
}

void propagateDeletions(vector<bool> &A_del, vector<bool> &A_ins,
    const vector<vector<int>> &children,
    vector<double> &Dist, vector<int> &Parent)
{
    queue<int> q;
    for (int i = 0; i < (int)A_del.size(); ++i)
        if (A_del[i]) q.push(i);
    while (!q.empty()) {
        int v = q.front(); q.pop();
        for (int c : children[v]) {
            if (!A_del[c]) {
                A_del[c] = A_ins[c] = true;
                Dist[c] = INF;
                Parent[c] = -1;
                q.push(c);
            }
        }
    }
}

void relaxAffected(vector<bool> &A_ins,
    vector<double> &Dist, vector<int> &Parent,
    const vector<vector<Edge>> &G)
{
    int n = Dist.size();
    bool changed;
    do {
        changed = false;
        vector<bool> next(n, false);
        for (int v = 0; v < n; ++v) if (A_ins[v]) {
            for (auto &e : G[v]) {
                int u = e.first; double w = e.second;
                if (Dist[u] > Dist[v] + w) {
                    Dist[u] = Dist[v] + w;
                    Parent[u] = v;
                    next[u] = changed = true;
                }
                if (Dist[v] > Dist[u] + w) {
                    Dist[v] = Dist[u] + w;
                    Parent[v] = u;
                    next[v] = changed = true;
                }
            }
        }
        for (int i = 0; i < n; ++i) if (next[i]) A_ins[i] = true;
    } while (changed);
}

void writeOutput(const string &f,
    const vector<double> &Dist, const vector<int> &Parent)
{
    ofstream out(f);
    out << "#vertex distance parent\n";
    for (int i = 0; i < (int)Dist.size(); ++i) {
        double d = Dist[i];
        out << i << " " << (d == INF ? -1.0 : d) << " " << Parent[i] << "\n";
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0]
             << " <graph> <sssp> <updates> <output>\n";
        return 1;
    }

    auto t_start = chrono::high_resolution_clock::now();
    int n;
    vector<vector<Edge>> G;
    readGraph(argv[1], n, G);
    auto t_after_graph = chrono::high_resolution_clock::now();

    vector<double> Dist;
    vector<int> Parent;
    readSSSP(argv[2], Dist, Parent, n);
    auto t_after_sssp = chrono::high_resolution_clock::now();

    vector<vector<int>> children(n);
    for (int v = 0; v < n; ++v)
        if (Parent[v] >= 0)
            children[Parent[v]].push_back(v);

    auto updates = readUpdates(argv[3]);
    auto t_after_read_updates = chrono::high_resolution_clock::now();

    vector<bool> A_del(n, false), A_ins(n, false);
    applyDeletions(updates, Dist, Parent, A_del, A_ins, children, G);
    auto t_after_deletions = chrono::high_resolution_clock::now();

    applyInsertions(updates, Dist, Parent, A_ins, G);
    auto t_after_insertions = chrono::high_resolution_clock::now();

    propagateDeletions(A_del, A_ins, children, Dist, Parent);
    auto t_after_prop = chrono::high_resolution_clock::now();

    relaxAffected(A_ins, Dist, Parent, G);
    auto t_after_relax = chrono::high_resolution_clock::now();

    writeOutput(argv[4], Dist, Parent);
    auto t_end = chrono::high_resolution_clock::now();

    auto to_sec = [&](auto t1, auto t2){
        return chrono::duration<double>(t2 - t1 ).count();
    };

    cout << "Timings (seconds):\n";
    cout << "  Graph read:       " << to_sec(t_start, t_after_graph) << "\n";
    cout << "  SSSP read:       " << to_sec(t_after_graph, t_after_sssp) << "\n";
    cout << "  Read updates:    " << to_sec(t_after_sssp, t_after_read_updates) << "\n";
    cout << "  Deletions:       " << to_sec(t_after_read_updates, t_after_deletions) << "\n";
    cout << "  Insertions:      " << to_sec(t_after_deletions, t_after_insertions) << "\n";
    cout << "  Propagate del:   " << to_sec(t_after_insertions, t_after_prop) << "\n";
    cout << "  Relax affected:  " << to_sec(t_after_prop, t_after_relax) << "\n";
    cout << "  Output write:    " << to_sec(t_after_relax, t_end) << "\n";
    cout << "  Total time:      " << to_sec(t_start, t_end) << "\n";

    return 0;
}
