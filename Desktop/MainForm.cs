using System.Diagnostics;
using System.ComponentModel;
using System.Drawing.Drawing2D;
using System.Net.Http.Json;
using System.Text.Json;

namespace TetrisStudio;

internal sealed class MainForm : Form
{
    private static readonly Color Bg = Color.FromArgb(235, 247, 255);
    private static readonly Color Card = Color.FromArgb(251, 253, 255);
    private static readonly Color Card2 = Color.FromArgb(226, 241, 254);
    private static readonly Color Edge = Color.FromArgb(191, 218, 242);
    private static readonly Color Muted = Color.FromArgb(73, 103, 137);
    private static readonly Color Ink = Color.FromArgb(22, 48, 77);
    private static readonly Color Purple = Color.FromArgb(42, 126, 225);
    private static readonly Color Mint = Color.FromArgb(13, 146, 190);
    private readonly HttpClient _http = new() { Timeout = TimeSpan.FromSeconds(30) };
    private readonly System.Windows.Forms.Timer _poll = new() { Interval = 90 };
    private readonly System.Windows.Forms.Timer _anim = new() { Interval = 34 };
    private readonly GameBoard _board = new();
    private readonly Label _score = new();
    private readonly Label _lines = new();
    private readonly Label _pieces = new();
    private readonly Label _objectiveValue = new();
    private readonly Label _objectiveDetail = new();
    private readonly Label _levelName = new();
    private readonly Label _status = new();
    private readonly PiecePreview _hold = new();
    private readonly PiecePreview[] _next = [new(), new(), new(), new()];
    private readonly Label _agentTag = new();
    private readonly ProgressLine _objective = new();
    private readonly DarkSelect _mode = new();
    private readonly DarkSelect _controller = new();
    private readonly DarkSelect _level = new();
    private readonly TrackBar _speed = new();
    private readonly Label _speedValue = new();
    private readonly Label _speedCaption = new();
    private int _agentSpeed = 1;
    private int _humanSpeed = 7;
    private bool _configuringSpeed;
    private readonly Button _start = new();
    private readonly Button _pause = new();
    private readonly Button _stop = new();
    private readonly Panel _canvasHost = new();
    private readonly Button[] _playButtons = new Button[6];
    private readonly string _demoRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));
    private Process? _backend;
    private Uri? _api;
    private JsonDocument? _state;
    private bool _requestBusy;
    private bool _closing;
    private float _pulse;
    private string _lastState = "menu";
    private readonly SemaphoreSlim _commands = new(1, 1);
    private int _pendingCommands;
    private int _stateEpoch;
    private readonly Label _holdNote = new();

    public MainForm()
    {
        Text = "Tetris Studio · RL Arcade";
        StartPosition = FormStartPosition.CenterScreen;
        MinimumSize = new Size(1120, 760);
        Size = new Size(1380, 980);
        AutoScaleMode = AutoScaleMode.Dpi;
        BackColor = Bg;
        ForeColor = Ink;
        Font = new Font("Segoe UI", 10F);
        KeyPreview = true;
        DoubleBuffered = true;
        BuildInterface();
        _poll.Tick += async (_, _) => await RefreshState();
        _anim.Tick += (_, _) => { _pulse += .04f; _board.Invalidate(); };
        _anim.Start();
        Load += async (_, _) => await StartBackend();
        FormClosing += (_, _) => Shutdown();
    }

    private void BuildInterface()
    {
        var root = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(24, 15, 24, 16), RowCount = 3, ColumnCount = 1, BackColor = Bg };
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 78));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
        Controls.Add(root);
        root.Controls.Add(BuildHeader(), 0, 0);

        var columns = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 3, RowCount = 1, Padding = new Padding(0, 12, 0, 12), BackColor = Bg };
        columns.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 310));
        columns.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        columns.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 300));
        root.Controls.Add(columns, 0, 1);
        columns.Controls.Add(BuildSettings(), 0, 0);
        columns.Controls.Add(BuildArena(), 1, 0);
        columns.Controls.Add(BuildStats(), 2, 0);
        root.Controls.Add(BuildFooter(), 0, 2);
    }

    private Control BuildHeader()
    {
        var bar = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 1, BackColor = Bg, Margin = Padding.Empty };
        bar.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 58));
        bar.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        bar.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        var mark = new Panel { Dock = DockStyle.Fill, Margin = new Padding(0, 10, 12, 12) };
        mark.Paint += (_, e) =>
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            using var brush = new SolidBrush(Purple);
            foreach (var cell in new[] { new Point(1, 0), new Point(0, 1), new Point(1, 1), new Point(2, 1) })
            {
                using var shape = Rounded(new RectangleF(2 + cell.X * 14, 6 + cell.Y * 14, 12, 12), 3);
                e.Graphics.FillPath(brush, shape);
            }
        };
        bar.Controls.Add(mark, 0, 0);
        var text = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 2, Margin = Padding.Empty };
        text.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        text.RowStyles.Add(new RowStyle(SizeType.Absolute, 43));
        text.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        var title = MakeLabel("BLOCK / LAB", 23, Ink, true); title.Dock = DockStyle.Fill; title.TextAlign = ContentAlignment.BottomLeft;
        var subtitle = MakeLabel("TETRIS  /  PLAY + DISCOVER", 9, Muted, false); subtitle.Dock = DockStyle.Fill; subtitle.TextAlign = ContentAlignment.MiddleLeft;
        text.Controls.Add(title, 0, 0); text.Controls.Add(subtitle, 0, 1); bar.Controls.Add(text, 1, 0);
        return bar;
    }

    private Control BuildSettings()
    {
        var stack = MakeStack();
        stack.Controls.Add(CardPanel("PLAY SETUP", BuildSetupContent()), 0, 0);
        stack.Controls.Add(CardPanel("AGENT LOADOUT", BuildAgentContent()), 0, 1);
        stack.Controls.Add(CardPanel("CONTROLS", BuildKeysContent()), 0, 2);
        return stack;
    }

    private Control BuildSetupContent()
    {
        var p = new TableLayoutPanel { Dock = DockStyle.Fill, BackColor = Card, ColumnCount = 1, RowCount = 10, Margin = Padding.Empty };
        int[] heights = [19, 38, 25, 38, 25, 38, 25, 40, 52, 44];
        foreach (int h in heights) p.RowStyles.Add(new RowStyle(SizeType.Absolute, h));
        void Caption(string text, int row) { var label = MakeLabel(text, 8, Muted, true); label.Dock = DockStyle.Fill; label.TextAlign = ContentAlignment.BottomLeft; label.Margin = new Padding(0, 0, 0, 4); p.Controls.Add(label, 0, row); }
        void Select(DarkSelect select, int row) { select.Dock = DockStyle.Fill; select.Margin = new Padding(0, 0, 0, 2); p.Controls.Add(select, 0, row); }
        Caption("GAME MODE", 0);
        _mode.Items.AddRange(["SURVIVAL RUN", "CHALLENGE LEVELS"]); _mode.SelectedIndex = 0; Select(_mode, 1);
        _mode.SelectedIndexChanged += (_, _) => { _level.Enabled = _mode.SelectedIndex == 1; UpdateAgentTag(); };
        Caption("WHO PLAYS", 2);
        _controller.Items.AddRange(["YOU · KEYBOARD", "DDQN AGENT"]); _controller.SelectedIndex = 0; Select(_controller, 3);
        _controller.SelectedIndexChanged += (_, _) => { UpdateAgentTag(); ConfigureSpeedControl(); };
        Caption("CHALLENGE LEVEL", 4);
        _level.Items.AddRange(["01  ·  FIRST WELL", "02  ·  DEEPER WELL", "03  ·  FIND THE WELL", "04  ·  SHIFT ONCE", "05  ·  LONG SHIFT", "06  ·  TWO SEGMENTS", "07  ·  THREE SEGMENTS", "08  ·  FINAL DESCENT"]);
        _level.SelectedIndex = 0; _level.Enabled = false; Select(_level, 5);
        _speedCaption.Text = "GAME SPEED  ·  SLOW → FAST"; _speedCaption.Font = new Font("Segoe UI", 8, FontStyle.Bold); _speedCaption.ForeColor = Muted;
        _speedCaption.Dock = DockStyle.Fill; _speedCaption.TextAlign = ContentAlignment.BottomLeft; _speedCaption.Margin = new Padding(0, 0, 0, 4); p.Controls.Add(_speedCaption, 0, 6);
        var speedRow = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, Margin = Padding.Empty };
        speedRow.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100)); speedRow.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 59));
        _speed.Minimum = 1; _speed.Maximum = 10; _speed.Value = 7; _speed.TickStyle = TickStyle.None;
        _speed.AutoSize = false; _speed.Dock = DockStyle.Fill; _speed.BackColor = Card; _speed.Margin = new Padding(0, 4, 6, 4); _speed.AccessibleName = "Game speed";
        _speed.SmallChange = _speed.LargeChange = 1;
        _speed.ValueChanged += async (_, _) =>
        {
            if (_configuringSpeed) return;
            bool agent = _controller.SelectedIndex == 1;
            if (agent) _agentSpeed = _speed.Value; else _humanSpeed = _speed.Value;
            UpdateSpeedText();
            // Setup selections take effect on Start. Only change playback live
            // when this speed control belongs to the controller of the current game.
            if (_state is not null && ReadString(_state.RootElement, "controller", "human") == (agent ? "agent" : "human"))
                await Post("/api/speed", agent ? (object)new { agentSpeed = _agentSpeed } : new { delay = CurrentDelay() });
        };
        _speedValue.Font = new Font("Consolas", 10, FontStyle.Bold); _speedValue.ForeColor = Purple; _speedValue.Dock = DockStyle.Fill; _speedValue.TextAlign = ContentAlignment.MiddleRight;
        speedRow.Controls.Add(_speed, 0, 0); speedRow.Controls.Add(_speedValue, 1, 0); p.Controls.Add(speedRow, 0, 7); UpdateSpeedText();
        _start.Text = "START GAME"; StyleButton(_start, Purple, Color.White); _start.Dock = DockStyle.Fill; _start.Margin = new Padding(0, 7, 0, 3); _start.Enabled = false;
        _start.Click += async (_, _) => await StartGame(); p.Controls.Add(_start, 0, 8);
        var actions = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, Margin = Padding.Empty };
        actions.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50)); actions.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
        _pause.Text = "PAUSE"; StyleButton(_pause, Card2, Ink); _pause.Dock = DockStyle.Fill; _pause.Margin = new Padding(0, 5, 5, 0); _pause.Enabled = false;
        _pause.Click += async (_, _) => await Post("/api/pause", new { }); actions.Controls.Add(_pause, 0, 0);
        _stop.Text = "STOP"; StyleButton(_stop, Card2, Ink); _stop.Dock = DockStyle.Fill; _stop.Margin = new Padding(5, 5, 0, 0); _stop.Enabled = false;
        _stop.Click += async (_, _) => await Post("/api/stop", new { }); actions.Controls.Add(_stop, 1, 0); p.Controls.Add(actions, 0, 9);
        return p;
    }

    private Control BuildAgentContent()
    {
        var p = new Panel { Dock = DockStyle.Fill, BackColor = Card };
        var orb = new AgentOrb { Location = new Point(0, 2), Size = new Size(40, 40) }; p.Controls.Add(orb);
        _agentTag.Text = "DDQN + DANGER"; _agentTag.Font = new Font("Consolas", 11, FontStyle.Bold); _agentTag.ForeColor = Ink; _agentTag.Location = new Point(50, 3); _agentTag.AutoSize = true; p.Controls.Add(_agentTag);
        var model = MakeLabel("FROZEN CHECKPOINT · INFERENCE", 8, Muted, false); model.Location = new Point(51, 25); model.AutoSize = true; p.Controls.Add(model);
        return p;
    }

    private Control BuildKeysContent()
    {
        var p = new Panel { Dock = DockStyle.Fill, BackColor = Card };
        string[] keys = ["←  →", "↓", "SPACE", "↑  /  X", "C", "P"];
        string[] actions = ["MOVE", "SOFT DROP", "HARD DROP", "ROTATE", "HOLD", "PAUSE"];
        for (int i = 0; i < keys.Length; i++)
        {
            int y = i * 26;
            var k = new Badge(keys[i], Muted) { Location = new Point(0, y), Size = new Size(84, 25) }; p.Controls.Add(k);
            var a = MakeLabel(actions[i], 8.7f, Muted, true); a.Location = new Point(101, y + 5); a.AutoSize = true; p.Controls.Add(a);
        }
        return p;
    }

    private Control BuildArena()
    {
        var frame = new Panel { Dock = DockStyle.Fill, Padding = new Padding(14, 12, 14, 13), BackColor = Card2 };
        frame.Paint += (_, e) => { using var pen = new Pen(Edge); e.Graphics.DrawRectangle(pen, 0, 0, frame.Width - 1, frame.Height - 1); };
        var layout = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 3, BackColor = Color.Transparent };
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 53)); layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100)); layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 50)); frame.Controls.Add(layout);
        var top = new Panel { Dock = DockStyle.Fill, BackColor = Color.Transparent };
        var heading = MakeLabel("THE ARENA", 12, Ink, true); heading.Location = new Point(0, 4); heading.AutoSize = true; top.Controls.Add(heading);
        _levelName.Text = "SURVIVAL · ENDLESS"; _levelName.Font = new Font("Consolas", 8.5f); _levelName.ForeColor = Mint; _levelName.Location = new Point(1, 27); _levelName.AutoSize = true; top.Controls.Add(_levelName);
        var live = new Badge("● LIVE VIEW", Mint) { Anchor = AnchorStyles.Top | AnchorStyles.Right, Size = new Size(105, 26), Location = new Point(frame.Width - 130, 6) }; top.Controls.Add(live);
        top.Resize += (_, _) => live.Left = top.ClientSize.Width - live.Width;
        layout.Controls.Add(top, 0, 0);
        _canvasHost.Dock = DockStyle.Fill; _canvasHost.BackColor = Color.FromArgb(207, 226, 245); _canvasHost.Padding = new Padding(12);
        _board.Dock = DockStyle.Fill; _canvasHost.Controls.Add(_board); layout.Controls.Add(_canvasHost, 0, 1);
        layout.Controls.Add(BuildPlayControls(), 0, 2);
        return frame;
    }

    private Control BuildPlayControls()
    {
        var row = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 6, RowCount = 1, BackColor = Color.Transparent, Padding = new Padding(0, 5, 0, 0) };
        for (int i = 0; i < 6; i++) row.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f / 6));
        string[] labels = ["◀  LEFT", "⟳  ROTATE", "▼  DOWN", "RIGHT  ▶", "HOLD  C", "DROP  SPACE"];
        string[] actions = ["left", "rotate", "down", "right", "hold", "drop"];
        for (int i = 0; i < labels.Length; i++)
        {
            int index = i;
            var button = new Button { Text = labels[i], Dock = DockStyle.Fill, Margin = new Padding(3, 0, 3, 0) };
            StyleButton(button, i == 5 ? Purple : Card2, i == 5 ? Color.White : Ink);
            button.Font = new Font("Segoe UI", 8.2f, FontStyle.Bold);
            button.Click += async (_, _) => await SendGameAction(actions[index]);
            _playButtons[i] = button; row.Controls.Add(button, i, 0);
        }
        UpdatePlayButtons();
        return row;
    }

    private Control BuildStats()
    {
        var stack = MakeStack();
        stack.Controls.Add(CardPanel("RUN TELEMETRY", BuildTelemetry()), 0, 0);
        stack.Controls.Add(CardPanel("HOLD BAY", BuildHold()), 0, 1);
        stack.Controls.Add(CardPanel("UP NEXT", BuildNext()), 0, 2);
        stack.Controls.Add(CardPanel("CHALLENGE OBJECTIVE", BuildObjective()), 0, 3);
        return stack;
    }

    private Control BuildTelemetry()
    {
        var p = new Panel { Dock = DockStyle.Fill, BackColor = Card };
        AddCaption(p, "SCORE", 0, 0); _score.Text = "000000"; _score.Font = new Font("Consolas", 27, FontStyle.Bold); _score.ForeColor = Purple; _score.Location = new Point(0, 18); _score.AutoSize = true; p.Controls.Add(_score);
        AddCaption(p, "LINES", 0, 69); _lines.Text = "00"; _lines.Font = new Font("Consolas", 18, FontStyle.Bold); _lines.ForeColor = Ink; _lines.Location = new Point(0, 86); _lines.AutoSize = true; p.Controls.Add(_lines);
        AddCaption(p, "PLACEMENTS", 151, 69); _pieces.Text = "000"; _pieces.Font = new Font("Consolas", 18, FontStyle.Bold); _pieces.ForeColor = Ink; _pieces.Location = new Point(151, 86); _pieces.AutoSize = true; p.Controls.Add(_pieces);
        return p;
    }

    private Control BuildHold()
    {
        var p = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, Margin = Padding.Empty };
        p.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 43)); p.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 57));
        _hold.Dock = DockStyle.Fill; _hold.EmptyText = "EMPTY"; _hold.Margin = new Padding(0, 0, 10, 0); p.Controls.Add(_hold, 0, 0);
        _holdNote.Text = "Press C or HOLD\nto save a piece."; _holdNote.Font = new Font("Segoe UI", 9); _holdNote.ForeColor = Muted; _holdNote.Dock = DockStyle.Fill; _holdNote.TextAlign = ContentAlignment.MiddleLeft;
        p.Controls.Add(_holdNote, 1, 0); return p;
    }

    private Control BuildNext()
    {
        var p = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 2, Margin = Padding.Empty };
        for (int i = 0; i < 2; i++) { p.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50)); p.RowStyles.Add(new RowStyle(SizeType.Percent, 50)); }
        int[] ids = [2, 3, 4, 7];
        for (int i = 0; i < _next.Length; i++) { _next[i].Dock = DockStyle.Fill; _next[i].Margin = new Padding(3); _next[i].SetSample(ids[i]); p.Controls.Add(_next[i], i % 2, i / 2); }
        return p;
    }

    private Control BuildObjective()
    {
        var p = new Panel { Dock = DockStyle.Fill, BackColor = Card };
        _objectiveValue.Text = "READY"; _objectiveValue.Font = new Font("Consolas", 16, FontStyle.Bold); _objectiveValue.ForeColor = Mint; _objectiveValue.Location = new Point(0, 0); _objectiveValue.AutoSize = true; p.Controls.Add(_objectiveValue);
        _objectiveDetail.Text = "Clear the original garbage cells"; _objectiveDetail.Font = new Font("Segoe UI", 9); _objectiveDetail.ForeColor = Muted; _objectiveDetail.Location = new Point(0, 29); _objectiveDetail.AutoSize = true; p.Controls.Add(_objectiveDetail);
        _objective.Value = 0; _objective.SetBounds(0, 55, 300, 10); p.Controls.Add(_objective);
        return p;
    }

    private Control BuildFooter()
    {
        var p = new Panel { Dock = DockStyle.Fill, BackColor = Bg };
        var sep = new Panel { Dock = DockStyle.Top, Height = 1, BackColor = Edge }; p.Controls.Add(sep);
        _status.Text = "SẴN SÀNG"; _status.Font = new Font("Consolas", 8.5f); _status.ForeColor = Muted; _status.Location = new Point(2, 14); _status.AutoSize = true; p.Controls.Add(_status);
        return p;
    }

    private static TableLayoutPanel MakeStack()
    {
        var t = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 0, Padding = new Padding(4, 0, 4, 0), BackColor = Bg, AutoScroll = true, GrowStyle = TableLayoutPanelGrowStyle.AddRows };
        t.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        return t;
    }

    private static Control CardPanel(string title, Control content)
    {
        int height = title switch { "PLAY SETUP" => 404, "AGENT LOADOUT" => 105, "CONTROLS" => 216, "RUN TELEMETRY" => 178, "HOLD BAY" => 146, "UP NEXT" => 190, _ => 146 };
        var panel = new TableLayoutPanel { Dock = DockStyle.Top, Height = height, MinimumSize = new Size(0, height), BackColor = Card, ColumnCount = 1, RowCount = 2, Padding = new Padding(15, 10, 15, 14), Margin = new Padding(0, 0, 0, 10) };
        panel.RowStyles.Add(new RowStyle(SizeType.Absolute, 29)); panel.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        var header = MakeLabel(title, 8.5f, Muted, true); header.Dock = DockStyle.Fill; header.Margin = Padding.Empty;
        header.Paint += (_, e) => { using var pen = new Pen(Edge); e.Graphics.DrawLine(pen, 0, header.Height - 7, header.Width, header.Height - 7); };
        panel.Paint += (_, e) => { using var pen = new Pen(Edge); e.Graphics.DrawRectangle(pen, 0, 0, panel.Width - 1, panel.Height - 1); };
        content.Dock = DockStyle.Fill; content.Margin = new Padding(0, 5, 0, 0);
        panel.Controls.Add(header, 0, 0); panel.Controls.Add(content, 0, 1); return panel;
    }

    private static Label MakeLabel(string text, float size, Color color, bool bold) => new() { Text = text, Font = new Font("Segoe UI", size, bold ? FontStyle.Bold : FontStyle.Regular), ForeColor = color, BackColor = Color.Transparent };
    private static void AddCaption(Control p, string text, int x, int y) { var l = MakeLabel(text, 8, Muted, true); l.Location = new Point(x, y); l.AutoSize = true; p.Controls.Add(l); }
    private static void StyleButton(Button b, Color background, Color foreground) { b.FlatStyle = FlatStyle.Flat; b.FlatAppearance.BorderColor = Edge; b.FlatAppearance.MouseOverBackColor = Color.FromArgb(Math.Min(255, background.R + 18), Math.Min(255, background.G + 18), Math.Min(255, background.B + 18)); b.BackColor = background; b.ForeColor = foreground; b.Font = new Font("Segoe UI", 10, FontStyle.Bold); b.Cursor = Cursors.Hand; }

    private void UpdateAgentTag() => _agentTag.Text = _mode.SelectedIndex == 1 ? "DDQN + SPARSE" : "DDQN + DANGER";

    private static GraphicsPath Rounded(RectangleF rect, float radius)
    {
        float d = radius * 2; var p = new GraphicsPath(); p.AddArc(rect.X, rect.Y, d, d, 180, 90); p.AddArc(rect.Right - d, rect.Y, d, d, 270, 90); p.AddArc(rect.Right - d, rect.Bottom - d, d, d, 0, 90); p.AddArc(rect.X, rect.Bottom - d, d, d, 90, 90); p.CloseFigure(); return p;
    }

    private async Task StartBackend()
    {
        try
        {
            string root = FindDemoRoot();
            string python = Path.Combine(root, "runtime", "python.exe");
            if (!File.Exists(python)) throw new FileNotFoundException("DEMO runtime not found", python);
            var start = new ProcessStartInfo(python, $"\"{Path.Combine(root, "demo_server.py")}\" --port 0") { WorkingDirectory = root, UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true };
            _backend = Process.Start(start) ?? throw new InvalidOperationException("Could not start the local game engine.");
            string? line = await _backend.StandardOutput.ReadLineAsync().WaitAsync(TimeSpan.FromSeconds(15));
            if (line is null || !line.StartsWith("DEMO ready: ", StringComparison.Ordinal)) throw new InvalidOperationException("Local game engine did not become ready. " + await _backend.StandardError.ReadToEndAsync());
            _api = new Uri(line[12..]);
            _status.Text = "SẴN SÀNG · CHỌN CHẾ ĐỘ ĐỂ BẮT ĐẦU";
            _start.Enabled = true;
            _poll.Start();
            await RefreshState();
        }
        catch (Exception ex) { _status.Text = "LỖI KHỞI ĐỘNG: " + ex.Message; _status.ForeColor = Color.Salmon; MessageBox.Show(this, ex.Message, "Không thể khởi động game", MessageBoxButtons.OK, MessageBoxIcon.Error); }
    }

    private string FindDemoRoot()
    {
        var path = new DirectoryInfo(AppContext.BaseDirectory);
        while (path is not null && !File.Exists(Path.Combine(path.FullName, "demo_server.py"))) path = path.Parent;
        return path?.FullName ?? _demoRoot;
    }

    private async Task StartGame()
    {
        if (_lastState == "level_clear" && _mode.SelectedIndex == 1 && _state is not null && _level.SelectedIndex == ReadInt(_state.RootElement, "levelIndex"))
            _level.SelectedIndex = (_level.SelectedIndex + 1) % _level.Items.Count;
        var payload = new { mode = _mode.SelectedIndex == 1 ? "challenge" : "survival", controller = _controller.SelectedIndex == 1 ? "agent" : "human", level = _level.SelectedIndex, delay = CurrentDelay(), agentSpeed = _agentSpeed };
        _start.Enabled = false;
        await Post("/api/start", payload);
        _start.Enabled = _api is not null;
        _board.Focus();
    }

    private async Task Post(string path, object data)
    {
        if (_api is null || _closing) return;
        _pendingCommands++; _stateEpoch++;
        await _commands.WaitAsync();
        try
        {
            using var doc = await GameApiClient.Post(_http, new Uri(_api, path), data);
            if (!_closing) SetState(doc.RootElement);
        }
        catch (Exception ex) { if (!_closing) { _status.Text = "LỖI ĐIỀU KHIỂN: " + ex.Message; _status.ForeColor = Color.Firebrick; } }
        finally { _pendingCommands--; _commands.Release(); }
    }

    private async Task RefreshState()
    {
        if (_api is null || _requestBusy || _closing || _pendingCommands > 0) return;
        int epoch = _stateEpoch;
        _requestBusy = true;
        try
        {
            using var doc = await _http.GetFromJsonAsync<JsonDocument>(new Uri(_api, "/api/state"));
            if (doc is not null && epoch == _stateEpoch && !_closing) SetState(doc.RootElement);
        }
        catch { _status.Text = "ĐANG KẾT NỐI LẠI GAME..."; _status.ForeColor = Color.FromArgb(179, 116, 15); }
        finally { _requestBusy = false; }
    }

    private void SetState(JsonElement root)
    {
        _state?.Dispose(); _state = JsonDocument.Parse(root.GetRawText());
        _board.State = _state.RootElement;
        string state = ReadString(root, "state", "menu");
        if (state != _lastState)
        {
            _lastState = state;
            string message = ReadString(root, "message", state);
            if (state == "error") message += ": " + ReadString(root, "error", "unknown error");
            _status.Text = message.ToUpperInvariant();
            _status.ForeColor = state switch { "playing" => Mint, "paused" => Color.FromArgb(179, 116, 15), "game_over" or "level_clear" => Purple, "error" => Color.Salmon, _ => Muted };
            _start.Text = state switch { "level_clear" when ReadString(root, "mode", "survival") == "challenge" => "NEXT LEVEL", "game_over" => "PLAY AGAIN", "playing" => "RESTART GAME", _ => "START GAME" };
        }
        _score.Text = ReadInt(root, "score").ToString("D6");
        _lines.Text = ReadInt(root, "lines").ToString("D2");
        _pieces.Text = ReadInt(root, "placements").ToString("D3");
        bool challenge = ReadString(root, "mode", "survival") == "challenge";
        _levelName.Text = challenge ? ReadString(root.GetProperty("level"), "label", "CHALLENGE") : "SURVIVAL · LEVEL " + ReadInt(root, "survivalLevel").ToString("00");
        _objectiveValue.Text = challenge ? $"{ReadInt(root, "remaining")} CELLS" : "ENDLESS RUN";
        _objectiveDetail.Text = challenge ? $"{ReadDouble(root, "removedPct"):0}% original garbage cleared" : "Survive as long as you can";
        _objective.Value = challenge ? Math.Clamp((int)ReadDouble(root, "removedPct"), 0, 100) : Math.Clamp(ReadInt(root, "lines") % 100, 0, 100);
        _hold.Piece = root.TryGetProperty("hold", out var hold) && hold.ValueKind == JsonValueKind.Object ? hold : default;
        int nextIndex = 0;
        if (root.TryGetProperty("queue", out var queue) && queue.ValueKind == JsonValueKind.Array)
            foreach (var piece in queue.EnumerateArray().Take(_next.Length)) _next[nextIndex++].Piece = piece;
        int[] sampleIds = [2, 3, 4, 7];
        for (; nextIndex < _next.Length; nextIndex++)
        {
            if (state == "menu") _next[nextIndex].SetSample(sampleIds[nextIndex]);
            else _next[nextIndex].Piece = default;
        }
        UpdateAgentTag();
        bool human = ReadString(root, "controller", "human") == "human";
        _holdNote.Text = !human ? "This agent uses\nplacements only."
            : hold.ValueKind == JsonValueKind.Object ? "Saved piece\nC / HOLD to swap" : "Press C or HOLD\nto save a piece.";
        _pause.Enabled = _stop.Enabled = state is "playing" or "paused";
        _pause.Text = state == "paused" ? "RESUME" : "PAUSE";
        UpdatePlayButtons();
        _board.Invalidate();
    }

    private async Task SendGameAction(string action)
    {
        if (action == "pause") await Post("/api/pause", new { });
        else await Post("/api/action", new { action });
    }

    private void UpdatePlayButtons()
    {
        bool canPlay = _lastState == "playing" && _state is not null && ReadString(_state.RootElement, "controller", "human") == "human";
        foreach (var button in _playButtons) if (button is not null) button.Enabled = canPlay;
    }

    private void UpdateSpeedText()
    {
        _speedValue.Text = _controller.SelectedIndex == 1 ? $"×{_agentSpeed}" : $"{CurrentDelay():0.00}s";
    }

    private void ConfigureSpeedControl()
    {
        _configuringSpeed = true;
        bool agent = _controller.SelectedIndex == 1;
        _speed.Maximum = agent ? 5 : 10;
        _speed.Value = agent ? _agentSpeed : _humanSpeed;
        _speedCaption.Text = agent ? "AGENT SPEED  ·  ×1 → ×5" : "GAME SPEED  ·  SLOW → FAST";
        _speed.AccessibleName = agent ? "Agent speed multiplier" : "Game speed";
        _configuringSpeed = false;
        UpdateSpeedText();
    }

    private double CurrentDelay() => 1.2 - (Math.Clamp(_humanSpeed, 1, 10) - 1) * (1.08 / 9.0);

    protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
    {
        if (_api is not null)
        {
            Keys key = keyData & Keys.KeyCode;
            string? action = key switch
            {
                Keys.Left => "left", Keys.Right => "right", Keys.Down => "down",
                Keys.Up or Keys.X => "rotate", Keys.Z => "rotate_ccw", Keys.C => "hold", Keys.Space => "drop",
                Keys.P or Keys.Escape => "pause", _ => null
            };
            if (action == "pause" && (_lastState is "playing" or "paused")) { _ = SendGameAction(action); return true; }
            if (action is not null && _lastState == "playing" && _state is not null && ReadString(_state.RootElement, "controller", "human") == "human" && !_speed.ContainsFocus && !_mode.ContainsFocus && !_controller.ContainsFocus && !_level.ContainsFocus) { _ = SendGameAction(action); return true; }
        }
        return base.ProcessCmdKey(ref msg, keyData);
    }

    private void Shutdown()
    {
        _closing = true; _poll.Stop(); _anim.Stop();
        try { _backend?.Kill(true); } catch { }
        _backend?.Dispose(); _http.Dispose(); _state?.Dispose();
    }

    private static string ReadString(JsonElement e, string key, string fallback) => e.ValueKind == JsonValueKind.Object && e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() ?? fallback : fallback;
    private static int ReadInt(JsonElement e, string key) => e.ValueKind == JsonValueKind.Object && e.TryGetProperty(key, out var v) && v.TryGetInt32(out int x) ? x : 0;
    private static double ReadDouble(JsonElement e, string key) => e.ValueKind == JsonValueKind.Object && e.TryGetProperty(key, out var v) && v.TryGetDouble(out double x) ? x : 0;

    private sealed class Badge : Control
    {
        private readonly Color _color;
        public Badge(string text, Color color) { Text = text; _color = color; Font = new Font("Consolas", 8, FontStyle.Bold); ForeColor = color; BackColor = Card2; }
        protected override void OnPaint(PaintEventArgs e) { base.OnPaint(e); e.Graphics.SmoothingMode = SmoothingMode.AntiAlias; using var path = Rounded(ClientRectangle, 5); using var fill = new SolidBrush(Color.FromArgb(24, _color)); using var pen = new Pen(Color.FromArgb(90, _color)); e.Graphics.FillPath(fill, path); e.Graphics.DrawPath(pen, path); TextRenderer.DrawText(e.Graphics, Text, Font, ClientRectangle, _color, TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter); }
    }

    private sealed class DarkSelect : UserControl
    {
        private readonly Button _button = new();
        private ContextMenuStrip? _menu;
        private int _selectedIndex = -1;
        public List<string> Items { get; } = [];
        public event EventHandler? SelectedIndexChanged;
        [Browsable(false), DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
        public int SelectedIndex
        {
            get => _selectedIndex;
            set
            {
                int next = Items.Count == 0 ? -1 : Math.Clamp(value, -1, Items.Count - 1);
                if (_selectedIndex == next) return;
                _selectedIndex = next; RefreshText(); SelectedIndexChanged?.Invoke(this, EventArgs.Empty);
            }
        }
        public DarkSelect()
        {
            AccessibleRole = AccessibleRole.ComboBox; BackColor = Card2; MinimumSize = new Size(120, 32); Height = 32; Font = new Font("Consolas", 9);
            _button.Dock = DockStyle.Fill; _button.FlatStyle = FlatStyle.Flat; _button.FlatAppearance.BorderSize = 0; _button.BackColor = Card2; _button.ForeColor = Ink; _button.Font = Font; _button.TextAlign = ContentAlignment.MiddleLeft; _button.Padding = new Padding(8, 0, 0, 0); _button.Cursor = Cursors.Hand; _button.Click += (_, _) => ShowOptions(); Controls.Add(_button);
            _button.Paint += (_, e) => { using var pen = new Pen(Edge); e.Graphics.DrawRectangle(pen, 0, 0, Width - 1, Height - 1); using var arrow = new SolidBrush(Muted); e.Graphics.FillPolygon(arrow, [new Point(Width - 18, 13), new Point(Width - 9, 13), new Point(Width - 13, 18)]); };
            Resize += (_, _) => _button.Padding = new Padding(8, 0, 22, 0);
            EnabledChanged += (_, _) => { _button.Enabled = Enabled; _button.ForeColor = Enabled ? Ink : Muted; Invalidate(); };
        }
        private void RefreshText() { _button.Text = _selectedIndex >= 0 ? Items[_selectedIndex] : "SELECT"; _button.AccessibleName = _button.Text; }
        private void ShowOptions()
        {
            if (!Enabled) return;
            _menu?.Dispose();
            var menu = new ContextMenuStrip { BackColor = Card2, ForeColor = Ink, ShowImageMargin = false, Font = new Font("Consolas", 9), Padding = new Padding(3), Renderer = new DarkMenuRenderer() };
            _menu = menu;
            for (int i = 0; i < Items.Count; i++)
            {
                int index = i; var item = new ToolStripMenuItem(Items[i]) { ForeColor = i == _selectedIndex ? Mint : Ink, BackColor = Card2, Padding = new Padding(7, 5, 7, 5) };
                item.Click += (_, _) => SelectedIndex = index; menu.Items.Add(item);
            }
            menu.Show(this, new Point(0, Height));
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) _menu?.Dispose();
            base.Dispose(disposing);
        }
    }

    private sealed class DarkMenuRenderer : ToolStripProfessionalRenderer
    {
        protected override void OnRenderMenuItemBackground(ToolStripItemRenderEventArgs e)
        {
            var rect = new Rectangle(Point.Empty, e.Item.Size); using var brush = new SolidBrush(e.Item.Selected ? Color.FromArgb(205, 229, 250) : Card2); e.Graphics.FillRectangle(brush, rect);
        }
        protected override void OnRenderToolStripBorder(ToolStripRenderEventArgs e) { using var pen = new Pen(Edge); e.Graphics.DrawRectangle(pen, 0, 0, e.ToolStrip.Width - 1, e.ToolStrip.Height - 1); }
    }

    private sealed class ProgressLine : Control
    {
        private int _value;
        [Browsable(false), DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
        public int Value { get => _value; set { _value = Math.Clamp(value, 0, 100); AccessibleName = $"Progress {_value}%"; Invalidate(); } }
        public ProgressLine() { AccessibleRole = AccessibleRole.ProgressBar; BackColor = Card2; }
        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e); e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            using var track = Rounded(new RectangleF(0, 0, Width - 1, Height - 1), Height / 2f); using var trackBrush = new SolidBrush(Card2); e.Graphics.FillPath(trackBrush, track);
            if (_value <= 0) return; float fillWidth = Math.Max(Height, (Width - 1) * _value / 100f); using var fill = Rounded(new RectangleF(0, 0, Math.Min(Width - 1, fillWidth), Height - 1), Height / 2f); using var fillBrush = new LinearGradientBrush(new Rectangle(0, 0, Math.Max(1, (int)fillWidth), Height), Mint, Color.FromArgb(49, 194, 255), LinearGradientMode.Horizontal); e.Graphics.FillPath(fillBrush, fill);
        }
    }

    private sealed class AgentOrb : Control
    {
        protected override void OnPaint(PaintEventArgs e) { base.OnPaint(e); e.Graphics.SmoothingMode = SmoothingMode.AntiAlias; using var b = new SolidBrush(Color.FromArgb(48, 42, 126, 225)); e.Graphics.FillEllipse(b, 2, 2, Width - 4, Height - 4); using var p = new Pen(Purple, 2); e.Graphics.DrawEllipse(p, 8, 8, Width - 16, Height - 16); using var d = new SolidBrush(Mint); e.Graphics.FillEllipse(d, Width - 17, 8, 7, 7); }
    }

    private sealed class PiecePreview : Control
    {
        private static readonly Color[] PieceColors = GameBoard.Colors;
        private JsonElement _piece;
        private int _sampleId = -1;
        [Browsable(false), DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
        public JsonElement Piece
        {
            get => _piece;
            set
            {
                // Snapshots are replaced and disposed on every poll; detach preview data
                // so a delayed WinForms paint cannot read from a disposed JsonDocument.
                _piece = value.ValueKind == JsonValueKind.Undefined ? default : value.Clone();
                _sampleId = -1;
                Invalidate();
            }
        }
        public string EmptyText = "";
        public PiecePreview() { DoubleBuffered = true; BackColor = Card2; }
        public void SetSample(int id) { _sampleId = id; _piece = default; Invalidate(); }
        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            if (_piece.ValueKind == JsonValueKind.Object && _piece.TryGetProperty("shape", out var shape) && shape.ValueKind == JsonValueKind.Array)
            {
                int id = Math.Clamp(ReadInt(_piece, "id"), 0, PieceColors.Length - 1);
                var cells = new List<Point>();
                for (int y = 0; y < shape.GetArrayLength(); y++) for (int x = 0; x < shape[y].GetArrayLength(); x++)
                    if (shape[y][x].GetInt32() != 0) cells.Add(new Point(x, y));
                if (cells.Count == 0) return;
                int minX = cells.Min(p => p.X), minY = cells.Min(p => p.Y);
                int cols = cells.Max(p => p.X) - minX + 1, rows = cells.Max(p => p.Y) - minY + 1;
                float tile = Math.Min(25, Math.Min((Width - 14f) / cols, (Height - 12f) / rows));
                float ox = (Width - cols * tile) / 2f, oy = (Height - rows * tile) / 2f;
                foreach (var cell in cells) GameBoard.DrawBlock(e.Graphics, ox + (cell.X - minX) * tile, oy + (cell.Y - minY) * tile, tile, PieceColors[id], false);
            }
            else if (_sampleId >= 0) DrawSample(e.Graphics, _sampleId);
            else TextRenderer.DrawText(e.Graphics, EmptyText, Font, ClientRectangle, Muted, TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter);
        }
        private void DrawSample(Graphics g, int id)
        {
            int[,] shape = id switch { 2 => new[,] { { 1, 1, 1, 1 } }, 3 => new[,] { { 1, 1 }, { 1, 1 } }, 4 => new[,] { { 0, 1, 0 }, { 1, 1, 1 } }, _ => new[,] { { 1, 0, 0 }, { 1, 1, 1 } } };
            int rows = shape.GetLength(0), cols = shape.GetLength(1); float tile = Math.Min(25, Math.Min((Width - 14f) / cols, (Height - 6f) / rows)); float ox = (Width - cols * tile) / 2f, oy = (Height - rows * tile) / 2f;
            for (int y = 0; y < rows; y++) for (int x = 0; x < cols; x++) if (shape[y, x] != 0) GameBoard.DrawBlock(g, ox + x * tile, oy + y * tile, tile, PieceColors[id], false);
        }
    }

    private sealed class GameBoard : Control
    {
        internal static readonly Color[] Colors = [Color.Transparent, Color.Transparent, Color.FromArgb(75, 216, 245), Color.FromArgb(255, 198, 82), Color.FromArgb(177, 130, 255), Color.FromArgb(83, 232, 178), Color.FromArgb(255, 103, 128), Color.FromArgb(101, 157, 255), Color.FromArgb(255, 153, 91)];
        private readonly float[] _stars = Enumerable.Range(0, 34).Select(i => (float)((i * 0.61803398875) % 1)).ToArray();
        private float _phase;
        [Browsable(false), DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
        public JsonElement State { get; set; }
        public GameBoard() { DoubleBuffered = true; BackColor = Color.FromArgb(12, 34, 59); TabStop = true; }
        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e); var g = e.Graphics; g.SmoothingMode = SmoothingMode.AntiAlias; g.Clear(BackColor);
            int boardW = 10, boardH = 20; float tile = Math.Min((Width - 45f) / boardW, (Height - 38f) / boardH); float bw = tile * boardW, bh = tile * boardH; float ox = (Width - bw) / 2f, oy = (Height - bh) / 2f;
            using (var bgBrush = new LinearGradientBrush(new RectangleF(ox, oy, bw, bh), Color.FromArgb(20, 57, 91), Color.FromArgb(10, 29, 52), 90f)) g.FillRectangle(bgBrush, ox, oy, bw, bh);
            DrawStars(g, ox, oy, bw, bh);
            using var grid = new Pen(Color.FromArgb(39, 132, 185, 230), Math.Max(1, tile * .022f));
            for (int x = 0; x <= boardW; x++) g.DrawLine(grid, ox + x * tile, oy, ox + x * tile, oy + bh);
            for (int y = 0; y <= boardH; y++) g.DrawLine(grid, ox, oy + y * tile, ox + bw, oy + y * tile);
            if (State.ValueKind == JsonValueKind.Object)
            {
                if (State.TryGetProperty("board", out var board) && board.ValueKind == JsonValueKind.Array)
                    for (int y = 0; y < boardH; y++) { var row = board[y]; for (int x = 0; x < boardW; x++) { int id = row[x].GetInt32(); if (id > 0) DrawBlock(g, ox + x * tile, oy + y * tile, tile, Colors[Math.Clamp(id, 0, 8)], State.TryGetProperty("provenance", out var prov) && prov.ValueKind == JsonValueKind.Array && prov[y][x].GetBoolean()); } }
                if (State.TryGetProperty("active", out var active) && active.ValueKind == JsonValueKind.Object)
                {
                    int id = ReadInt(active, "id"); Color color = Colors[Math.Clamp(id, 0, 8)]; int ax = ReadInt(State, "activeX"), ay = ReadInt(State, "activeY"), gy = ReadInt(State, "ghostY");
                    if (active.TryGetProperty("shape", out var shape) && shape.ValueKind == JsonValueKind.Array)
                    {
                        DrawPiece(g, shape, ax, gy, tile, ox, oy, color, true);
                        DrawPiece(g, shape, ax, ay, tile, ox, oy, color, false);
                    }
                }
                string state = ReadString(State, "state", "menu");
                if (state is "paused" or "game_over" or "level_clear" or "error") DrawOverlay(g, state, ox, oy, bw, bh);
                else if (state == "menu") DrawWelcome(g, ox, oy, bw, bh);
            }
            else DrawWelcome(g, ox, oy, bw, bh);
            using var border = new Pen(Color.FromArgb(145, 131, 185, 225), 1); g.DrawRectangle(border, ox, oy, bw, bh);
        }
        private void DrawStars(Graphics g, float ox, float oy, float bw, float bh)
        {
            _phase += .0018f;
            for (int i = 0; i < _stars.Length; i++) { float x = ox + ((i * .371f + _phase) % 1f) * bw; float y = oy + ((i * .719f) % 1f) * bh; int alpha = 18 + (int)(30 * (0.5 + 0.5 * Math.Sin(_phase * 15 + i))); using var b = new SolidBrush(Color.FromArgb(alpha, 128, 178, 228)); g.FillEllipse(b, x, y, 2.2f, 2.2f); }
        }
        private static void DrawPiece(Graphics g, JsonElement shape, int x, int y, float tile, float ox, float oy, Color c, bool ghost)
        {
            for (int r = 0; r < shape.GetArrayLength(); r++) for (int col = 0; col < shape[r].GetArrayLength(); col++) if (shape[r][col].GetInt32() != 0) { int yy = y + r; if (yy < 0 || yy >= 20 || x + col < 0 || x + col >= 10) continue; DrawBlock(g, ox + (x + col) * tile, oy + yy * tile, tile, c, false, ghost); }
        }
        internal static void DrawBlock(Graphics g, float x, float y, float tile, Color color, bool garbage, bool ghost = false)
        {
            float inset = tile * .075f; var rect = new RectangleF(x + inset, y + inset, tile - inset * 2, tile - inset * 2);
            using var path = Rounded(rect, tile * .11f);
            if (ghost) { using var gp = new Pen(Color.FromArgb(115, color), Math.Max(1, tile * .07f)); g.DrawPath(gp, path); return; }
            using var shadow = new SolidBrush(Color.FromArgb(75, 0, 0, 0)); g.FillPath(shadow, Rounded(new RectangleF(rect.X + 1, rect.Y + 2, rect.Width, rect.Height), tile * .11f));
            using var fill = new LinearGradientBrush(rect, ControlPaint.Light(color, .24f), ControlPaint.Dark(color, .11f), LinearGradientMode.Vertical); g.FillPath(fill, path);
            using var outline = new Pen(Color.FromArgb(175, 255, 255, 255), Math.Max(1, tile * .035f)); g.DrawPath(outline, path);
            using var shine = new Pen(Color.FromArgb(115, 255, 255, 255), Math.Max(1, tile * .045f)); g.DrawLine(shine, rect.X + tile * .12f, rect.Y + tile * .14f, rect.Right - tile * .12f, rect.Y + tile * .14f);
            if (garbage) { using var gpen = new Pen(Color.FromArgb(180, 255, 232, 185), Math.Max(1, tile * .055f)); g.DrawRectangle(gpen, rect.X + tile * .12f, rect.Y + tile * .12f, rect.Width - tile * .24f, rect.Height - tile * .24f); }
        }
        private static void DrawWelcome(Graphics g, float x, float y, float w, float h)
        {
            using var title = new Font("Segoe UI", 16, FontStyle.Bold); using var small = new Font("Consolas", 8.5f); using var white = new SolidBrush(Color.FromArgb(238, 245, 255)); using var accent = new SolidBrush(Purple); using var muted = new SolidBrush(Muted);
            var card = new RectangleF(x + 10, y + h * .43f - 8, w - 20, 105); using var path = Rounded(card, 10); using var panel = new SolidBrush(Color.FromArgb(225, 12, 20, 35)); using var edge = new Pen(Color.FromArgb(100, 132, 151, 190)); g.FillPath(panel, path); g.DrawPath(edge, path);
            using var centered = new StringFormat { Alignment = StringAlignment.Center };
            g.DrawString("READY PLAYER ONE", title, white, new RectangleF(card.X + 4, card.Y + 14, card.Width - 8, 32), centered);
            g.DrawString("CHOOSE A MODE · PRESS START", small, accent, new RectangleF(card.X + 5, card.Y + 54, card.Width - 10, 17), centered);
            g.DrawString("YOUR NEXT MOVE STARTS HERE", small, muted, new RectangleF(card.X + 5, card.Y + 76, card.Width - 10, 16), centered);
        }
        private static void DrawOverlay(Graphics g, string state, float x, float y, float w, float h)
        {
            using var shade = new SolidBrush(Color.FromArgb(188, 5, 9, 18)); g.FillRectangle(shade, x, y, w, h);
            string text = state switch { "paused" => "PAUSED", "level_clear" => "LEVEL CLEAR!", "game_over" => "GAME OVER", _ => "ENGINE ERROR" };
            using var title = new Font("Segoe UI", 27, FontStyle.Bold); using var hint = new Font("Consolas", 10); using var a = new SolidBrush(state == "level_clear" ? Color.Aquamarine : Color.White); using var b = new SolidBrush(Color.LightSteelBlue);
            var r = new RectangleF(x, y + h * .43f, w, 44); g.DrawString(text, title, a, r, new StringFormat { Alignment = StringAlignment.Center });
            var r2 = new RectangleF(x, y + h * .43f + 52, w, 22); g.DrawString(state == "paused" ? "PRESS P TO RESUME" : "PRESS START GAME TO PLAY AGAIN", hint, b, r2, new StringFormat { Alignment = StringAlignment.Center });
        }
    }
}
