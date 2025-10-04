import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Calendar } from "./ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "./ui/popover";
import { Badge } from "./ui/badge";
import { Play, Square, Calendar as CalendarIcon, TrendingUp, BarChart3, MessageSquare } from "lucide-react";
import { format } from "date-fns";

interface ControlPanelProps {
  ticker: string;
  setTicker: (ticker: string) => void;
  timeframe: string;
  setTimeframe: (timeframe: string) => void;
  selectedDate?: Date;
  setSelectedDate: (date?: Date) => void;
  isAnalyzing: boolean;
  onStartAnalysis: () => void;
  onStopAnalysis: () => void;
  analysisId: string | null;
  messageCount: number;
  chartCount: number;
}

const timeframes = ["1D", "5D", "1M", "3M", "6M", "1Y", "YTD", "MAX"];

export const ControlPanel = ({
  ticker,
  setTicker,
  timeframe,
  setTimeframe,
  selectedDate,
  setSelectedDate,
  isAnalyzing,
  onStartAnalysis,
  onStopAnalysis,
  analysisId,
  messageCount,
  chartCount,
}: ControlPanelProps) => {
  return (
    <div className="border-b border-border bg-card">
      <div className="p-4">
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <Input
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="Enter ticker (e.g. AAPL)"
            className="w-32 uppercase"
            maxLength={10}
            disabled={isAnalyzing}
          />

          <Select value={timeframe} onValueChange={setTimeframe} disabled={isAnalyzing}>
            <SelectTrigger className="w-24">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {timeframes.map((tf) => (
                <SelectItem key={tf} value={tf}>
                  {tf}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" disabled={isAnalyzing} className="gap-2">
                <CalendarIcon className="h-4 w-4" />
                {selectedDate ? format(selectedDate, "PPP") : "Optional Date"}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <Calendar
                mode="single"
                selected={selectedDate}
                onSelect={setSelectedDate}
                initialFocus
              />
            </PopoverContent>
          </Popover>

          {!isAnalyzing ? (
            <Button onClick={onStartAnalysis} className="gap-2">
              <Play className="h-4 w-4" />
              Start Demo Analysis
            </Button>
          ) : (
            <Button onClick={onStopAnalysis} variant="destructive" className="gap-2">
              <Square className="h-4 w-4" />
              Stop
            </Button>
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary" className="gap-1">
            <TrendingUp className="h-3 w-3" />
            Ticker: {ticker || "N/A"}
          </Badge>
          <Badge variant="secondary" className="gap-1">
            <BarChart3 className="h-3 w-3" />
            Timeframe: {timeframe}
          </Badge>
          <Badge variant="secondary" className="gap-1">
            <MessageSquare className="h-3 w-3" />
            Messages: {messageCount}
          </Badge>
          <Badge variant="secondary" className="gap-1">
            <BarChart3 className="h-3 w-3" />
            Charts: {chartCount}
          </Badge>
          {analysisId && (
            <Badge variant="secondary" className="gap-1">
              Run ID: {analysisId.slice(-8)}
            </Badge>
          )}
        </div>
      </div>
    </div>
  );
};
